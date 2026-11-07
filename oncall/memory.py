"""Hindsight memory wrapper: retain / recall / reflect, plus a labelled offline fallback.

Real backend
------------
`HindsightMemory` talks to Hindsight (Cloud or self-hosted) with the official
`hindsight-client`. Every incident is retained as one self-contained document
(`document_id=incident-INC-014`), so re-seeding is idempotent. Fix outcomes are
retained separately (`document_id=outcome-INC-014-restart_service`), so an
engineer clicking Worked twice does not create two memories.

Retrieval quality comes from the query we build: alert + symptoms + a slice of
the real log lines, with the service and its dependency names. Facts are grouped
back into incident-level memories using the metadata Hindsight returns with
every recalled fact.

Offline fallback
----------------
If Hindsight is unreachable (or no URL is configured), `LocalMemory` keeps the
demo usable with BM25-ish keyword retrieval over a local JSON store. It is
clearly labelled in the UI and never presented as Hindsight.
"""
from __future__ import annotations

import asyncio
import json
import math
import queue
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from oncall.config import Settings
from oncall.schemas import (
    FIX_LABELS,
    FixTried,
    Incident,
    MemoryHit,
    OutcomeRecord,
)


class _SdkRunner:
    """Run every Hindsight SDK call on one thread with one event loop.

    The SDK's sync wrappers do `asyncio.get_event_loop().run_until_complete(...)`.
    Streamlit executes each script run in a fresh thread, so those calls land on a
    different loop every time and aiohttp fails with "Timeout context manager
    should be used inside a task" — which broke outcome writes and memory stats.
    One owned thread, one owned loop, calls serialized.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], dict[str, Any]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._body, name="hindsight-sdk", daemon=True)
        self._thread.start()

    def _body(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            job = self._jobs.get()
            if job is None:
                break
            fn, box = job
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc
            finally:
                box["done"].set()

    def call(self, fn: Callable[[], Any], timeout: float = 120.0) -> Any:
        box: dict[str, Any] = {"done": threading.Event(), "value": None, "error": None}
        self._jobs.put((fn, box))
        if not box["done"].wait(timeout):
            raise MemoryUnavailable(f"Hindsight call did not return within {timeout:.0f}s")
        if box["error"] is not None:
            raise box["error"]
        return box["value"]


_runner: _SdkRunner | None = None
_runner_lock = threading.Lock()


def _get_runner() -> _SdkRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = _SdkRunner()
    return _runner

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are", "has",
    "have", "had", "not", "but", "after", "before", "into", "over", "when", "then",
    "than", "its", "it's", "our", "their", "they", "them", "which", "while", "also",
    "can", "could", "would", "should", "will", "been", "being", "does", "did", "get",
    "got", "out", "off", "per", "via", "vs", "a", "an", "of", "in", "on", "at", "to",
    "is", "as", "by", "or", "we", "you", "i", "be", "up", "no", "min", "ms", "s",
}

INCIDENT_MISSION = (
    "I am the on-call memory for a production engineering team. Remember every incident: "
    "the service, its symptoms, the root cause, and which fixes worked, failed or only "
    "partially worked. Prefer specific, evidence-backed fixes over generic restarts, and "
    "say when a fix that once worked has failed since."
)


class MemoryUnavailable(RuntimeError):
    """Raised when the memory backend cannot be reached."""


class MemoryBackend(Protocol):
    name: str
    bank_id: str
    degraded: bool

    def ensure_bank(self) -> None: ...
    def retain_incident(self, incident: Incident | dict) -> str: ...
    def retain_outcome(self, record: OutcomeRecord) -> str: ...
    def recall_similar(self, alert: str, service: str, symptoms: list[str],
                       logs_excerpt: str, k: int = 5,
                       as_of: datetime | None = None) -> list[MemoryHit]: ...
    def reflect_pattern(self, service_or_topic: str) -> str: ...
    def stats(self) -> dict[str, Any]: ...
    def delete_bank(self) -> None: ...


# --------------------------------------------------------------------------
# Memory text + query construction (shared by both backends)
# --------------------------------------------------------------------------
def _as_incident(incident: Incident | dict) -> Incident:
    return incident if isinstance(incident, Incident) else Incident.model_validate(incident)


def _fix_sentence(fix: FixTried) -> str:
    label = FIX_LABELS.get(fix.fix, fix.fix)
    note = f" ({fix.note})" if fix.note else ""
    return f"{fix.fix} [{label}] -> {fix.outcome.upper()}{note}"


def incident_memory_text(incident: Incident | dict) -> str:
    inc = _as_incident(incident)
    when = inc.opened_at.strftime("%Y-%m-%d")
    symptoms = "; ".join(inc.symptoms) or inc.alert
    fixes = "; ".join(_fix_sentence(f) for f in inc.fixes_tried) or "none recorded"
    resolved = f"Resolved in {inc.resolution_minutes} min"
    if inc.resolved_by:
        resolved += f" by {inc.resolved_by}"
    return (
        f"Incident {inc.id} ({inc.service}, {when}).\n"
        f"Alert: {inc.alert}.\n"
        f"Symptom: {symptoms}.\n"
        f"Root cause: {inc.root_cause}\n"
        f"Fixes: {fixes}.\n"
        f"{resolved}."
    )


def outcome_memory_text(record: OutcomeRecord) -> str:
    label = FIX_LABELS.get(record.fix, record.fix)
    note = f" Note: {record.note}" if record.note else ""
    alert = f"Alert: {record.alert}.\n" if record.alert else ""
    return (
        f"Fix outcome for incident {record.incident_id} ({record.service}, {record.family}).\n"
        f"{alert}"
        f"Fix: {record.fix} [{label}] -> {record.outcome.upper()}.{note}\n"
        f"Recorded by the on-call engineer on {record.recorded_at.strftime('%Y-%m-%d')}."
    )


def outcomes_metadata(incident: Incident | dict) -> str:
    inc = _as_incident(incident)
    return ";".join(f"{f.fix}:{f.outcome}" for f in inc.fixes_tried)


def build_recall_query(alert: str, service: str, symptoms: Iterable[str],
                       logs_excerpt: str) -> str:
    """A descriptive retrieval query; Hindsight rejects queries over 500 tokens."""
    excerpt = " ".join(logs_excerpt.split())
    if len(excerpt) > 900:
        excerpt = excerpt[:900].rsplit(" ", 1)[0]
    symptom_text = "; ".join(s for s in symptoms if s)[:400]
    query = (
        f"Production incident on {service}: {alert}. "
        f"Symptoms: {symptom_text}. "
        f"Log evidence: {excerpt} "
        f"Which past incident had the same failure pattern, what was its root cause, "
        f"and which fixes worked or failed?"
    )
    return query[:1900]


def _score_of(raw: Any) -> float | None:
    if isinstance(raw, dict):
        values = [float(v) for v in raw.values() if isinstance(v, (int, float))]
        return max(values) if values else None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# --------------------------------------------------------------------------
# Hindsight backend
# --------------------------------------------------------------------------
class HindsightMemory:
    name = "hindsight"
    degraded = False

    def __init__(self, settings: Settings, bank_id: str | None = None,
                 client: Any | None = None) -> None:
        if not settings.hindsight_base_url and client is None:
            raise MemoryUnavailable("HINDSIGHT_BASE_URL is not set")
        try:
            from hindsight_client import Hindsight
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise MemoryUnavailable("hindsight-client is not installed") from exc

        self.bank_id = bank_id or settings.hindsight_bank_id
        self._timeout = float(settings.hindsight_timeout)
        self._runner = _get_runner()
        self._client = client if client is not None else self._runner.call(
            lambda: Hindsight(
                base_url=settings.hindsight_base_url,
                api_key=settings.hindsight_api_key or None,
                timeout=settings.hindsight_timeout,
            ))
        self._bank_ready = False
        self._lock = threading.Lock()

    def _call(self, fn: Callable[[], Any]) -> Any:
        return self._runner.call(fn, timeout=self._timeout + 30)

    # -- setup ------------------------------------------------------------
    def ensure_bank(self) -> None:
        if self._bank_ready:
            return
        with self._lock:
            if self._bank_ready:
                return
            try:
                self._call(lambda: self._client.get_bank_config(self.bank_id))
                self._bank_ready = True
                return
            except Exception as exc:  # noqa: BLE001 - 404 means "create it"
                status = getattr(exc, "status", None)
                if status not in (404, None):
                    raise MemoryUnavailable(f"cannot read bank {self.bank_id!r}: {exc}") from exc
            try:
                self._call(lambda: self._client.create_bank(
                    bank_id=self.bank_id,
                    name="OnCall Memory",
                    mission=INCIDENT_MISSION,
                    disposition={"skepticism": 4, "literalism": 4, "empathy": 2},
                ))
            except Exception as exc:  # noqa: BLE001 - concurrent creation / exists
                message = str(exc).lower()
                if "exist" not in message and "409" not in message:
                    raise MemoryUnavailable(f"cannot create bank {self.bank_id!r}: {exc}") from exc
            self._bank_ready = True

    # -- writes -----------------------------------------------------------
    def retain_incident(self, incident: Incident | dict) -> str:
        inc = _as_incident(incident)
        self.ensure_bank()
        try:
            self._call(lambda: self._client.retain(
                bank_id=self.bank_id,
                content=incident_memory_text(inc),
                context="incident postmortem",
                timestamp=inc.opened_at,
                document_id=f"incident-{inc.id}",
                metadata={
                    "record_type": "incident",
                    "incident_id": inc.id,
                    "service": inc.service,
                    "family": inc.family,
                    "outcomes": outcomes_metadata(inc),
                    "source": "oncall-memory",
                },
            ))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(f"retain failed for {inc.id}: {exc}") from exc
        return f"incident-{inc.id}"

    def retain_outcome(self, record: OutcomeRecord) -> str:
        self.ensure_bank()
        document_id = f"outcome-{record.incident_id}-{record.fix}"
        try:
            self._call(lambda: self._client.retain(
                bank_id=self.bank_id,
                content=outcome_memory_text(record),
                context="incident fix outcome",
                timestamp=record.recorded_at,
                document_id=document_id,
                metadata={
                    "record_type": "outcome",
                    "incident_id": record.incident_id,
                    "service": record.service,
                    "family": record.family,
                    "alert": record.alert,
                    "fix": record.fix,
                    "outcome": record.outcome,
                    "source": "oncall-memory",
                },
            ))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(f"retain outcome failed for {record.incident_id}: {exc}") from exc
        return document_id

    # -- reads ------------------------------------------------------------
    def recall_similar(self, alert: str, service: str, symptoms: list[str],
                       logs_excerpt: str, k: int = 5,
                       as_of: datetime | None = None) -> list[MemoryHit]:
        self.ensure_bank()
        query = build_recall_query(alert, service, symptoms, logs_excerpt)
        try:
            # Raw facts (world/experience), not consolidated observations: an
            # observation comes back with metadata=None and no document_id, so
            # it cannot be attributed to an incident or filtered by pattern.
            # Provenance beats consolidation for triage.
            response = self._call(lambda: self._client.recall(
                bank_id=self.bank_id,
                query=query,
                types=["world", "experience"],
                budget="mid",
                max_tokens=1400,
                query_timestamp=(as_of if isinstance(as_of, str) else as_of.isoformat())
                if as_of else None,
            ))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(f"recall failed: {exc}") from exc
        results = list(getattr(response, "results", []) or [])

        grouped: dict[str, dict[str, Any]] = {}
        for result in results:
            meta = {str(k2): str(v) for k2, v in (getattr(result, "metadata", None) or {}).items()}
            record_type = meta.get("record_type", "fact")
            incident_id = meta.get("incident_id", "")
            group_key = (f"outcome|{incident_id}|{meta.get('fix', '')}"
                         if record_type == "outcome"
                         else meta.get("document_id") or getattr(result, "document_id", None)
                         or incident_id or getattr(result, "id", "fact"))
            group = grouped.setdefault(group_key, {
                "texts": [], "score": None, "metadata": {}, "occurred": None,
            })
            text = (getattr(result, "text", "") or "").strip()
            if text and text not in group["texts"]:
                group["texts"].append(text)
            score = _score_of(getattr(result, "scores", None))
            if score is not None and (group["score"] is None or score > group["score"]):
                group["score"] = score
            for key, value in meta.items():
                group["metadata"].setdefault(key, value)
            group["occurred"] = group["occurred"] or _iso(
                getattr(result, "occurred_start", None) or getattr(result, "mentioned_at", None)
            )

        hits: list[MemoryHit] = []
        for group in grouped.values():
            texts = group["texts"][:6]
            body = " ".join(t.replace("\n", " ").strip() for t in texts)
            if len(body) > 1600:
                body = body[:1600].rsplit(" ", 1)[0] + "..."
            meta = group["metadata"]
            header = ""
            if meta.get("incident_id"):
                date = (group["occurred"] or "")[:10]
                header = f"{meta['incident_id']} ({meta.get('service', service)}, {date}) — "
            hits.append(MemoryHit(
                id=str(group.get("id") or meta.get("document_id") or meta.get("incident_id", "")),
                text=header + body,
                score=group["score"],
                metadata=meta,
                occurred=group["occurred"],
            ))
        hits.sort(key=lambda h: (h.score is not None, h.score or 0.0), reverse=True)
        return hits[:k]

    def reflect_pattern(self, service_or_topic: str) -> str:
        self.ensure_bank()
        query = (
            f"What do we know about {service_or_topic}? Summarise recurring failure patterns, "
            f"root causes, and which fixes worked or failed. Cite incident IDs."
        )
        try:
            response = self._call(lambda: self._client.reflect(
                bank_id=self.bank_id,
                query=query,
                budget="mid",
                context="on-call incident history review",
            ))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(f"reflect failed: {exc}") from exc
        return (getattr(response, "text", "") or "").strip()

    def stats(self) -> dict[str, Any]:
        self.ensure_bank()
        try:
            page = self._call(lambda: self._client.list_memories(bank_id=self.bank_id, limit=1000))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(f"stats failed: {exc}") from exc
        items = list(getattr(page, "items", []) or [])
        incidents: set[str] = set()
        outcomes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for item in items:
            meta = getattr(item, "metadata", None) or {}
            if meta.get("record_type") == "incident" and meta.get("incident_id"):
                incidents.add(str(meta["incident_id"]))
            if meta.get("record_type") == "outcome" and meta.get("fix"):
                key = (str(meta.get("service", "?")), str(meta["fix"]))
                outcomes[key][str(meta.get("outcome", "unknown"))] += 1
        reliable = [
            {"service": svc, "fix": fix, "worked": c["worked"], "failed": c["failed"],
             "partial": c["partial"]}
            for (svc, fix), c in outcomes.items()
        ]
        reliable.sort(key=lambda r: (r["worked"] - r["failed"], r["worked"]), reverse=True)
        return {
            "backend": self.name,
            "bank_id": self.bank_id,
            "memories": int(getattr(page, "total", len(items)) or len(items)),
            "incidents": len(incidents),
            "outcomes": sum(sum(c.values()) for c in outcomes.values()),
            "reliable_fixes": reliable[:5],
        }

    def delete_bank(self) -> None:
        try:
            self._call(lambda: self._client.delete_bank(self.bank_id))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(f"delete bank failed: {exc}") from exc
        self._bank_ready = False

    def close(self) -> None:
        try:
            self._runner.call(lambda: self._client.close(), timeout=30.0)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Offline fallback (clearly labelled, never presented as Hindsight)
# --------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) >= 3 and t not in STOPWORDS]


class LocalMemory:
    """Keyword retrieval over a local JSON file. Demo-only degraded mode."""

    name = "local-keyword"
    degraded = True

    def __init__(self, store_dir: Path, bank_id: str) -> None:
        self.bank_id = bank_id
        self._dir = store_dir
        self._path = store_dir / f"{bank_id}.json"
        self._lock = threading.Lock()
        self._docs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._docs = {d["id"]: d for d in raw.get("docs", [])}
            except (json.JSONDecodeError, KeyError):
                self._docs = {}

    def _flush(self) -> None:
        payload = {"bank_id": self.bank_id, "docs": list(self._docs.values())}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ensure_bank(self) -> None:
        if not self._path.exists():
            self._flush()

    def _upsert(self, doc_id: str, text: str, metadata: dict[str, str],
                timestamp: datetime) -> str:
        with self._lock:
            self._docs[doc_id] = {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            }
            self._flush()
        return doc_id

    def retain_incident(self, incident: Incident | dict) -> str:
        inc = _as_incident(incident)
        return self._upsert(
            f"incident-{inc.id}", incident_memory_text(inc),
            {
                "record_type": "incident",
                "incident_id": inc.id,
                "service": inc.service,
                "family": inc.family,
                "outcomes": outcomes_metadata(inc),
                "source": "oncall-memory",
            },
            inc.opened_at,
        )

    def retain_outcome(self, record: OutcomeRecord) -> str:
        return self._upsert(
            f"outcome-{record.incident_id}-{record.fix}", outcome_memory_text(record),
            {
                "record_type": "outcome",
                "incident_id": record.incident_id,
                "service": record.service,
                "family": record.family,
                "alert": record.alert,
                "fix": record.fix,
                "outcome": record.outcome,
                "source": "oncall-memory",
            },
            record.recorded_at,
        )

    def recall_similar(self, alert: str, service: str, symptoms: list[str],
                       logs_excerpt: str, k: int = 5,
                       as_of: datetime | None = None) -> list[MemoryHit]:
        if not self._docs:
            return []
        query = build_recall_query(alert, service, symptoms, logs_excerpt)
        q_tokens = Counter(_tokens(query))
        if not q_tokens:
            return []
        doc_tokens = {doc_id: _tokens(d["text"]) for doc_id, d in self._docs.items()}
        df: Counter = Counter()
        for tokens in doc_tokens.values():
            df.update(set(tokens))
        n_docs = max(1, len(doc_tokens))
        idf = {t: math.log((n_docs + 1) / (count + 0.5)) for t, count in df.items()}

        def cosine(tokens: list[str]) -> float:
            if not tokens:
                return 0.0
            tf = Counter(tokens)
            dot = 0.0
            for token, q_count in q_tokens.items():
                if token in tf:
                    weight = idf.get(token, 1.0)
                    dot += q_count * tf[token] * weight * weight
            q_norm = math.sqrt(sum((c * idf.get(t, 1.0)) ** 2 for t, c in q_tokens.items()))
            d_norm = math.sqrt(sum((c * idf.get(t, 1.0)) ** 2 for t, c in tf.items()))
            return dot / (q_norm * d_norm) if q_norm and d_norm else 0.0

        scored = []
        for doc_id, tokens in doc_tokens.items():
            score = cosine(tokens)
            meta = self._docs[doc_id]["metadata"]
            if meta.get("service") == service:
                score += 0.05
            if meta.get("record_type") == "incident":
                score += 0.02
            scored.append((score, doc_id))
        scored.sort(reverse=True)

        # Absolute floor: a novel incident must be able to come back empty.
        top = scored[0][0] if scored else 0.0
        floor = max(0.16, top * 0.45)
        hits: list[MemoryHit] = []
        for score, doc_id in scored:
            if score < floor or len(hits) >= k:
                break
            doc = self._docs[doc_id]
            text = doc["text"]
            if len(text) > 1800:
                text = text[:1800].rsplit(" ", 1)[0] + "..."
            hits.append(MemoryHit(
                id=doc_id,
                text=text,
                score=round(score, 4),
                metadata=doc["metadata"],
                occurred=doc.get("timestamp"),
            ))
        return hits

    def reflect_pattern(self, service_or_topic: str) -> str:
        topic = service_or_topic.lower()
        matches = [d for d in self._docs.values()
                   if topic in d["text"].lower() or topic in d["metadata"].get("service", "").lower()]
        incidents = [d for d in matches if d["metadata"].get("record_type") == "incident"]
        if not incidents:
            return f"No recorded incidents mention {service_or_topic}."
        worked: Counter = Counter()
        failed: Counter = Counter()
        for doc in incidents[-12:]:
            for pair in doc["metadata"].get("outcomes", "").split(";"):
                if ":" not in pair:
                    continue
                fix, outcome = pair.split(":", 1)
                (worked if outcome == "worked" else failed)[fix] += 1
        lines = [f"Local keyword summary for {service_or_topic} ({len(incidents)} recorded incidents)."]
        for fix, count in worked.most_common(3):
            lines.append(f"Worked: {fix} ({count}/{count + failed[fix]}).")
        for fix, count in failed.most_common(3):
            if not worked[fix]:
                lines.append(f"Failed: {fix} ({count} time(s)).")
        lines.append(f"Most recent: {incidents[-1]['text'].splitlines()[0]}")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        incidents = [d for d in self._docs.values() if d["metadata"].get("record_type") == "incident"]
        outcome_docs = [d for d in self._docs.values() if d["metadata"].get("record_type") == "outcome"]
        outcomes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for doc in outcome_docs:
            meta = doc["metadata"]
            outcomes[(meta.get("service", "?"), meta.get("fix", "?"))][meta.get("outcome", "?")] += 1
        reliable = [
            {"service": svc, "fix": fix, "worked": c["worked"], "failed": c["failed"],
             "partial": c["partial"]}
            for (svc, fix), c in outcomes.items()
        ]
        reliable.sort(key=lambda r: (r["worked"] - r["failed"], r["worked"]), reverse=True)
        return {
            "backend": self.name,
            "bank_id": self.bank_id,
            "memories": len(self._docs),
            "incidents": len(incidents),
            "outcomes": len(outcome_docs),
            "reliable_fixes": reliable[:5],
        }

    def delete_bank(self) -> None:
        with self._lock:
            self._docs = {}
            if self._path.exists():
                self._path.unlink()

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Factory + helpers
# --------------------------------------------------------------------------
def build_memory(settings: Settings, bank_id: str | None = None,
                 force_local: bool = False) -> tuple[MemoryBackend, str | None]:
    """Return (backend, warning). Falls back to local keyword memory in `auto` mode."""
    bank = bank_id or settings.hindsight_bank_id
    want_local = force_local or settings.memory_backend == "local"

    def local(reason: str) -> tuple[MemoryBackend, str]:
        return LocalMemory(settings.base_dir / ".memory_store", bank), reason

    if want_local:
        return local("Using the local keyword fallback (MEMORY_BACKEND=local). Not Hindsight.")
    if not settings.hindsight_base_url:
        return local(
            "HINDSIGHT_BASE_URL is not set — using the local keyword fallback. "
            "Add your Hindsight URL and key to .env for real memory."
        )
    try:
        backend = HindsightMemory(settings, bank_id=bank)
        backend.ensure_bank()
        return backend, None
    except MemoryUnavailable as exc:
        if settings.memory_backend == "hindsight":
            raise
        return local(f"Hindsight unavailable ({exc}). Using local keyword fallback.")


def fix_history_from_memories(memories: Iterable[MemoryHit]) -> dict[str, dict[str, Any]]:
    """Aggregate recorded outcomes from recalled memory metadata.

    Metadata is the reliable channel: Hindsight may rephrase the retained text,
    but the `outcomes` and `fix`/`outcome` metadata fields survive verbatim.
    """
    history: dict[str, dict[str, Any]] = {}

    def add(fix: str, outcome: str, incident_id: str = "", service: str = "") -> None:
        if not fix:
            return
        entry = history.setdefault(fix, {"worked": 0, "failed": 0, "partial": 0,
                                         "incidents": set(), "services": set()})
        if outcome in entry:
            entry[outcome] += 1
        if incident_id:
            entry["incidents"].add(incident_id)
        if service:
            entry["services"].add(service)

    for hit in memories:
        meta = hit.metadata or {}
        for pair in str(meta.get("outcomes", "")).split(";"):
            if ":" not in pair:
                continue
            fix, outcome = pair.split(":", 1)
            add(fix.strip(), outcome.strip(), meta.get("incident_id", ""), meta.get("service", ""))
        if meta.get("record_type") == "outcome":
            add(meta.get("fix", ""), meta.get("outcome", ""),
                meta.get("incident_id", ""), meta.get("service", ""))
    return history


def past_outcomes_text(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "none"
    worked, failed, partial = entry["worked"], entry["failed"], entry["partial"]
    if not (worked or failed or partial):
        return "none"
    return f"worked {worked}/{worked + failed + partial} | failed {failed} | partial {partial}"


def conflicting_fixes(history: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(fix for fix, e in history.items() if e["worked"] and e["failed"])


def memory_incident_ids(memories: Iterable[MemoryHit]) -> set[str]:
    ids: set[str] = set()
    for hit in memories:
        incident_id = (hit.metadata or {}).get("incident_id")
        if incident_id:
            ids.add(incident_id)
        ids.update(re.findall(r"INC-\d{3,4}|LIVE-\d{8}-\d{6}", hit.text))
    return ids
