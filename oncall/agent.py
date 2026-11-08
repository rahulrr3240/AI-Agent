"""Incident triage pipeline.

v1 (default) is deterministic in *tool* order — logs, deploys, metrics, memory —
but the reasoning step is the LLM. If Groq is unavailable or returns garbage
after every retry, a deterministic heuristic analyst answers instead, so the
demo never dies on stage. The Investigation records exactly which path ran.

Memory is the differentiator, not decoration: recalled outcomes are converted
into an explicit per-fix record (`worked/failed/partial`) that is given to the
model, and a small guardrail keeps fixes with recorded failures below fixes with
recorded successes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from oncall.config import Settings
from oncall.llm import GroqLLM, LLMError, schema_hint
from oncall.live import LiveEvidence
from oncall.memory import (
    MemoryBackend,
    MemoryUnavailable,
    build_memory,
    conflicting_fixes,
    fix_history_from_memories,
    memory_incident_ids,
    past_outcomes_text,
)
from oncall.schemas import (
    FIX_LABELS,
    AgentResult,
    EvalIncident,
    Incident,
    Investigation,
    MemoryHit,
    OutcomeRecord,
    RankedFix,
    SimilarIncident,
    StepEvent,
)
from oncall.tools import SimulatedWorld, detect_pattern, tool_descriptions

# The reasoning call's completion budget. Groq counts `prompt + max_tokens`
# against its tokens-per-minute limit, so the benchmark paces on this number.
REASONING_MAX_TOKENS = 1200

SYSTEM_PROMPT = """You are OnCall Memory, an incident-response copilot for a production engineering team.

Rules:
1. Use ONLY the supplied incident context (logs, deploys, metrics, recalled memories). Do not invent services, versions, log lines or incident IDs.
2. Cite specific evidence: exact log lines, deploy versions/times or metric values.
3. Recorded fix outcomes are advisory tallies. A recalled memory that records a team rule, a constraint, or the reason a fix was blocked or failed overrides the tallies when the current incident's deploy, service or logs match that context; cite its incident id and say so in the rationale. Prefer same-service outcomes over other services. If a fix failed for the same service and failure pattern, demote it below fixes that worked there.
4. If a fix worked once and failed later, mention the conflict and lower confidence.
5. If the recalled memories are not genuinely similar to this incident, set no_relevant_history=true and give a clearly generic best-effort answer with confidence "low". Do not cite any incident IDs in that case.
6. Only cite incident IDs that appear in the recalled memories (they look like INC-###, or LIVE-… for incidents resolved from real alerts). Never cite the incident you are investigating. Put the reason each one is similar in why_similar.
7. ranked_fixes use ONLY ids from the supplied FIX VOCABULARY, ordered best first.
8. The answer is advisory: no action is executed.
9. Key order is part of the contract: write `ranked_fixes` immediately after `confidence`, then `no_relevant_history`, before `evidence` and `similar_incidents`. A response without ranked_fixes is invalid and will be rejected, so never stop before it.

Return one JSON object matching the requested schema. No prose outside JSON."""


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


@dataclass
class StepTracker:
    emit: Callable[[StepEvent], None] | None = None
    steps: list[StepEvent] = field(default_factory=list)

    def add(self, name: str, label: str, status: str = "done", detail: str = "",
            duration_ms: int = 0) -> StepEvent:
        event = StepEvent(name=name, label=label, status=status, detail=detail,
                          duration_ms=duration_ms)
        self.steps.append(event)
        if self.emit:
            self.emit(event)
        return event

    def run(self, name: str, label: str, fn: Callable[[], Any],
            describe: Callable[[Any], str]) -> Any:
        self.add(name, label, status="running")
        started = time.perf_counter()
        value = fn()
        duration = int((time.perf_counter() - started) * 1000)
        event = self.steps[-1]
        event.status = "done"
        event.detail = describe(value)
        event.duration_ms = duration
        if self.emit:
            self.emit(event)
        return value

    def skip(self, name: str, label: str, detail: str) -> None:
        self.add(name, label, status="skipped", detail=detail)


def _generic_fixes() -> list[str]:
    return ["restart_service", "rollback_deploy", "scale_up"]


def _root_cause_from_memory(hits: list[MemoryHit]) -> str | None:
    for hit in hits:
        text = hit.text
        marker = "Root cause:"
        if marker in text:
            snippet = text.split(marker, 1)[1]
            snippet = snippet.split("Fixes:")[0].split("\n")[0].strip().rstrip(".")
            if snippet and len(snippet) > 8:
                return snippet
    return None


def _error_lines(log_text: str, limit: int = 3) -> list[str]:
    lines = [line.strip() for line in log_text.splitlines()
             if "ERROR" in line or "level=error" in line or "OOMKilled" in line]
    return [line[:200] for line in lines[:limit]]


def _minutes_before(reference: datetime, value: str) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int((reference - parsed).total_seconds() // 60)


def _supplied_deploys(deploys: list[dict], as_of: datetime) -> list[dict]:
    """Deploy records the engineer pasted, annotated with their distance to the alert."""
    out = []
    for deploy in deploys:
        item = dict(deploy)
        item["minutes_before_alert"] = _minutes_before(as_of, item.get("deployed_at", ""))
        out.append(item)
    out.sort(key=lambda d: str(d.get("deployed_at", "")), reverse=True)
    return out


def _deploy_line(deploy: dict) -> str:
    minutes = deploy.get("minutes_before_alert")
    timing = f"{minutes}m before alert" if minutes is not None else "deploy time not supplied"
    return (f"{deploy['service']} {deploy['version']} at {deploy.get('deployed_at') or '?'} "
            f"({timing})")


def _metrics_line(metrics: dict) -> str:
    summary = metrics.get("summary") or {}
    parts = []
    if summary.get("peak_p95_ms") is not None:
        parts.append(f"peak p95 {summary['peak_p95_ms']}ms")
    if summary.get("peak_error_rate_pct") is not None:
        parts.append(f"peak error {summary['peak_error_rate_pct']}%")
    return ", ".join(parts) or "snapshot loaded without a latency/error summary"


class InvestigationAgent:
    def __init__(self, settings: Settings, world: SimulatedWorld | None = None,
                 memory: MemoryBackend | None = None, llm: GroqLLM | None = None,
                 memory_warning: str | None = None,
                 allow_fallback: bool = True) -> None:
        """`allow_fallback=False` makes a Groq failure an error instead of quietly
        answering with the deterministic heuristic. The benchmark uses it so its
        headline numbers can only contain rows the LLM actually answered."""
        self.settings = settings
        self.world = world or SimulatedWorld(settings.data_dir)
        if memory is None:
            memory, detected_warning = build_memory(settings)
            memory_warning = memory_warning or detected_warning
        self.memory = memory
        self.memory_warning = memory_warning
        self.llm = llm or GroqLLM(settings)
        self.allow_fallback = allow_fallback

    # ------------------------------------------------------------------ core
    def investigate(self, incident: Incident | EvalIncident | dict,
                    memory_enabled: bool = True,
                    emit: Callable[[StepEvent], None] | None = None,
                    evidence: LiveEvidence | None = None) -> Investigation:
        """Run one investigation. `evidence` switches the tools from the simulated
        world to what an engineer supplied for a live alert."""
        if isinstance(incident, str):
            found = self.world.incident(incident)
            if found is None:
                raise ValueError(f"unknown incident {incident}")
            incident = found
        inc = incident if isinstance(incident, Incident) else Incident.model_validate(incident)
        tracker = StepTracker(emit)
        warning = self.memory_warning
        effective_memory = memory_enabled

        logs, deploys, metrics, log_excerpt = self._collect(inc, evidence, tracker)

        detected, _pattern_score = detect_pattern(
            f"{inc.alert} {' '.join(inc.symptoms)} {logs.get('text', '')}",
            self.world.families(),
        )
        memories: list[MemoryHit] = []
        if not memory_enabled:
            tracker.skip("memory", "Recalling similar incidents", "Memory is OFF — recall skipped (control run)")
        else:
            try:
                memories = tracker.run(
                    "memory", "Recalling similar incidents",
                    lambda: self.memory.recall_similar(
                        alert=inc.alert, service=inc.service, symptoms=inc.symptoms,
                        logs_excerpt=log_excerpt, k=8, as_of=inc.opened_at,
                    ),
                    lambda r: (f"{len(r)} recalled memor{'y' if len(r) == 1 else 'ies'}"
                               + (" (local keyword fallback)" if self.memory.degraded else "")
                               + (f"; pattern: {detected}" if detected else "")),
                )
                if detected:
                    # Precision gate: a remembered incident from a different
                    # failure family is not evidence, however similar it sounds.
                    memories = [m for m in memories if m.metadata.get("family") == detected]
                    tracker.steps[-1].detail += f" — {len(memories)} in same pattern family"
                    if emit:
                        emit(tracker.steps[-1])
            except MemoryUnavailable as exc:
                warning = f"Hindsight recall failed: {exc}. Continuing with memory OFF behavior."
                effective_memory = False
                tracker.add("memory", "Recalling similar incidents", status="error",
                            detail="Hindsight unavailable — falling back to no-history answer")

        if self.settings.agent_mode == "agentic" and self.llm.configured and effective_memory:
            result, model, latency, fallback_reason, memories, tokens = self._agentic(
                inc, logs, deploys, metrics, memories, memory_enabled, tracker
            )
        else:
            result, model, latency, fallback_reason, tokens = self._reason(
                inc, logs, deploys, metrics, memories, effective_memory, tracker
            )

        history = fix_history_from_memories(memories) if effective_memory else {}
        conflicts = conflicting_fixes(history) if effective_memory else []
        result = self._postprocess(result, memories, history, effective_memory, conflicts,
                                   inc.service)
        result.similar_incidents = [s for s in result.similar_incidents if s.id != inc.id]

        if evidence is not None:
            # A live alert must not look better-evidenced than it is.
            for line in evidence.gaps():
                if line not in result.evidence:
                    result.evidence.append(line)
            if not evidence.has_logs:
                result.confidence = "low"

        if not logs.get("missing") and logs.get("line_count", 0) == 0:
            result.evidence.append("Log extract was empty — evidence is limited to metrics and deploys.")
            result.confidence = "low"

        return Investigation(
            incident_id=inc.id,
            memory_enabled=memory_enabled,
            detected_pattern=detected,
            result=result,
            steps=tracker.steps,
            memories=memories,
            memory_backend=self.memory.name,
            memory_warning=warning,
            llm_model=model,
            llm_latency_ms=latency,
            llm_prompt_tokens=tokens["prompt"],
            llm_completion_tokens=tokens["completion"],
            used_deterministic_fallback=bool(fallback_reason),
            fallback_reason=fallback_reason,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------ evidence
    def _collect(self, inc: Incident, evidence: LiveEvidence | None,
                 tracker: StepTracker) -> tuple[dict, list[dict], dict, str]:
        """Read logs, deploys and metrics, then return the log excerpt for recall.

        With `evidence` the tools read what the engineer supplied for a real
        alert; without it they read the simulated world, exactly as before.
        """
        if evidence is not None:
            logs = tracker.run(
                "logs", "Reading supplied logs",
                lambda: evidence.log_dict(),
                lambda r: (f"{r['line_count']} pasted lines, {r['error_lines']} errors"
                           if not r["missing"] else "no logs were supplied with this alert"),
            )
            deploys = tracker.run(
                "deploys", "Checking supplied deploys",
                lambda: _supplied_deploys(evidence.deploys, inc.opened_at),
                lambda r: (f"{len(r)} supplied deploy(s): " + ", ".join(_deploy_line(d) for d in r[:2])
                           if r else "no deploy history was supplied"),
            )
            metrics = tracker.run(
                "metrics", "Reading supplied metrics",
                lambda: evidence.metrics_dict(),
                lambda r: (_metrics_line(r) if r.get("available") else "no metrics were supplied"),
            )
            excerpt = " ".join(evidence.logs_text.split())[-900:]
            return logs, deploys, metrics, excerpt

        logs = tracker.run(
            "logs", "Reading logs",
            lambda: self.world.get_logs(inc.service, inc.id),
            lambda r: (f"{r['line_count']} lines, {r['error_lines']} errors"
                       if not r["missing"] else "log extract unavailable"),
        )
        deploys = tracker.run(
            "deploys", "Checking recent deploys",
            lambda: self.world.get_recent_deploys(inc.service, hours=6, as_of=inc.opened_at),
            lambda r: (f"{len(r)} deploy(s): " + ", ".join(
                f"{d['service']} {d['version']} ({d['minutes_before_alert']}m before)" for d in r[:2])
                if r else "no deploys in the last 6h"),
        )
        metrics = tracker.run(
            "metrics", "Reading metrics",
            lambda: self.world.get_metrics(inc.service, inc.id),
            lambda r: (f"peak p95 {r['summary']['peak_p95_ms']}ms, "
                       f"peak error {r['summary']['peak_error_rate_pct']}%"
                       if r.get("available") else "metrics snapshot unavailable"),
        )
        return logs, deploys, metrics, self.world.log_excerpt(inc.id)

    # --------------------------------------------------------------- reasoning
    def _reason(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                memories: list[MemoryHit], memory_on: bool,
                tracker: StepTracker) -> tuple[AgentResult, str, int, str | None, dict[str, int]]:
        started = time.perf_counter()
        tracker.add("reasoning", "Reasoning over evidence", status="running")
        no_tokens = {"prompt": 0, "completion": 0}

        if not self.llm.configured:
            if not self.allow_fallback:
                tracker.add("reasoning", "Reasoning over evidence", status="error",
                            detail="Groq key not configured; fallback is disabled")
                raise LLMError("GROQ_API_KEY is not set")
            result = self._heuristic(inc, logs, deploys, metrics, memories, memory_on)
            duration = int((time.perf_counter() - started) * 1000)
            tracker.add("reasoning", "Reasoning over evidence",
                        status="done",
                        detail="Groq key not configured — deterministic heuristic analyst",
                        duration_ms=duration)
            return result, "heuristic (no GROQ_API_KEY)", duration, "GROQ_API_KEY is not set", no_tokens

        messages = self._build_messages(inc, logs, deploys, metrics, memories, memory_on)
        try:
            llm_result = self.llm.chat_json(messages, AgentResult,
                                            max_tokens=REASONING_MAX_TOKENS)
        except LLMError as exc:
            if not self.allow_fallback:
                tracker.add("reasoning", "Reasoning over evidence", status="error",
                            detail="Groq failed after retries; fallback is disabled")
                raise
            result = self._heuristic(inc, logs, deploys, metrics, memories, memory_on)
            duration = int((time.perf_counter() - started) * 1000)
            tracker.add("reasoning", "Reasoning over evidence", status="done",
                        detail="Groq failed after retries — deterministic heuristic used",
                        duration_ms=duration)
            return result, "heuristic (Groq failed)", duration, str(exc), no_tokens

        duration = int((time.perf_counter() - started) * 1000)
        detail = f"{llm_result.model} · {llm_result.latency_ms}ms · {llm_result.attempts} attempt(s)"
        if llm_result.notes:
            detail += " · " + "; ".join(llm_result.notes)
        tracker.add("reasoning", "Reasoning over evidence", status="done", detail=detail,
                    duration_ms=duration)
        return (llm_result.data, llm_result.model, llm_result.latency_ms, None,
                {"prompt": llm_result.prompt_tokens, "completion": llm_result.completion_tokens})

    def _build_messages(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                        memories: list[MemoryHit], memory_on: bool) -> list[dict[str, str]]:
        history = fix_history_from_memories(memories) if memory_on else {}
        def scoped(entry: dict[str, Any], service: str) -> dict[str, Any] | None:
            """Outcomes recorded for this incident's own service, if any."""
            by_service = (entry.get("by_service") or {}).get(service)
            if not by_service:
                return None
            return {
                "service": service,
                "worked": by_service["worked"],
                "failed": by_service["failed"],
                "partial": by_service["partial"],
                "incidents": sorted(by_service["incidents"]),
            }

        history_rows = [
            {
                "fix": fix,
                "same_service": scoped(entry, inc.service),
                "family_wide": {
                    "worked": entry["worked"],
                    "failed": entry["failed"],
                    "partial": entry["partial"],
                    "services": sorted(entry["services"]),
                    "incidents": sorted(entry["incidents"]),
                },
            }
            for fix, entry in sorted(
                history.items(), key=lambda kv: (kv[1]["worked"] - kv[1]["failed"]), reverse=True)
        ]
        memory_block = [
            {
                "incident_id": hit.metadata.get("incident_id"),
                "record_type": hit.metadata.get("record_type"),
                "service": hit.metadata.get("service"),
                "occurred": hit.occurred,
                "text": hit.text,
            }
            for hit in memories
        ]
        context = {
            "incident": {
                "id": inc.id,
                "service": inc.service,
                "alert": inc.alert,
                "opened_at": _iso(inc.opened_at),
                "symptoms": inc.symptoms,
                "severity": inc.severity,
            },
            "logs": logs.get("text", "")[-3000:],
            "recent_deploys": deploys,
            "metrics_summary": metrics.get("summary"),
            "metrics_series": metrics.get("series", [])[-9:],
            "memory_enabled": memory_on,
            "recalled_memories": memory_block,
            "recorded_fix_outcomes": history_rows,
            "fix_vocabulary": FIX_LABELS,
        }
        user = (
            "Investigate this alert and return the JSON object.\n\n"
            + json.dumps(context, separators=(",", ":"))
            + "\n\nSchema (all fields required):\n" + schema_hint(AgentResult)
            + "\nWrite ranked_fixes immediately after confidence, before no_relevant_history, evidence and similar_incidents."
            + "\nranked_fixes: 1-5 entries, unique ids taken from FIX VOCABULARY, best first."
            + " A response without ranked_fixes is invalid."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------- heuristic
    def _heuristic(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                   memories: list[MemoryHit], memory_on: bool) -> AgentResult:
        history = fix_history_from_memories(memories) if memory_on else {}
        evidence: list[str] = _error_lines(logs.get("text", ""))
        if deploys:
            first = deploys[0]
            evidence.append(
                f"Deploy {first['id']}: {first['service']} {first['version']} at "
                f"{first['deployed_at']} ({first['minutes_before_alert']}m before alert)"
            )
        if metrics.get("available"):
            evidence.append(f"Metrics: {_metrics_line(metrics)}")
        if not evidence:  # evidence is required and non-empty in AgentResult
            evidence.append("The log extract had no error lines and no deploy or metric context was available.")

        similar: list[SimilarIncident] = []
        seen: set[str] = set()
        for hit in memories:
            incident_id = hit.metadata.get("incident_id", "")
            if incident_id and incident_id not in seen:
                seen.add(incident_id)
                similar.append(SimilarIncident(
                    id=incident_id,
                    why_similar="same failure signature in the recalled memory",
                ))
            if len(similar) == 3:
                break

        ranked: list[RankedFix] = []
        if history:
            def same(entry: dict[str, Any]) -> dict[str, Any]:
                return (entry.get("by_service") or {}).get(inc.service) or {}

            ordered = sorted(
                history.items(),
                key=lambda kv: (same(kv[1]).get("worked", 0) > 0,
                                same(kv[1]).get("worked", 0) - same(kv[1]).get("failed", 0),
                                kv[1]["worked"] > 0,
                                kv[1]["worked"] - kv[1]["failed"],
                                kv[1]["worked"]),
                reverse=True,
            )
            for fix, entry in ordered[:4]:
                if entry["worked"] == 0 and same(entry).get("worked", 0) == 0:
                    continue
                mine = same(entry)
                ranked.append(RankedFix(
                    fix=fix,
                    rationale=(f"Recorded {mine.get('worked', 0)} success(es) and "
                               f"{mine.get('failed', 0)} failure(s) on {inc.service}; "
                               f"{entry['worked']} success(es) and {entry['failed']} failure(s) "
                               f"across all services."),
                    based_on_history=True,
                    past_outcomes=past_outcomes_text(entry, inc.service),
                ))
        for fix in _generic_fixes():
            if len(ranked) >= 3:
                break
            if all(r.fix != fix for r in ranked):
                ranked.append(RankedFix(
                    fix=fix,
                    rationale="Generic best-effort step; no matching history was consulted.",
                    based_on_history=False,
                    past_outcomes=past_outcomes_text(history.get(fix), inc.service),
                ))
        if not ranked:  # pragma: no cover - generic list is never empty
            ranked = [RankedFix(fix="restart_service", rationale="Best effort.", based_on_history=False)]

        if memory_on and similar:
            recorded_cause = _root_cause_from_memory(memories)
            cause = (
                f"Closest recorded incident {similar[0].id} shows this failure pattern. "
                f"Recorded root cause: {recorded_cause}"
                if recorded_cause else
                f"Failure pattern matches {similar[0].id}; its recorded root cause is the most likely explanation."
            )
            confidence = "high" if history and ranked[0].based_on_history and ranked[0].past_outcomes.startswith("worked") else "medium"
            no_history = False
        else:
            cause = (
                "Root cause not determined. The service is degraded (see evidence) but there is no "
                "relevant historical incident to compare against; this is a generic best-effort triage."
            )
            confidence = "low"
            no_history = bool(memory_on)
            similar = []

        return AgentResult(
            likely_cause=cause,
            confidence=confidence,
            evidence=evidence[:5],
            similar_incidents=similar,
            ranked_fixes=ranked[:5],
            no_relevant_history=no_history,
        )

    # ----------------------------------------------------------- postprocess
    def _postprocess(self, result: AgentResult, memories: list[MemoryHit],
                     history: dict[str, dict[str, Any]], memory_on: bool,
                     conflicts: list[str], service: str = "") -> AgentResult:
        valid_ids = memory_incident_ids(memories) if memory_on else set()
        cleaned_similar = []
        seen: set[str] = set()
        for item in result.similar_incidents:
            if not memory_on:
                break
            if item.id in valid_ids and item.id not in seen:
                seen.add(item.id)
                cleaned_similar.append(item)
        result.similar_incidents = cleaned_similar

        if not memory_on:
            result.no_relevant_history = False
            history = {}

        if memory_on and not memories:
            result.no_relevant_history = True
            result.similar_incidents = []
            result.confidence = "low"

        if result.no_relevant_history:
            result.similar_incidents = []
            result.confidence = "low"

        for fix in result.ranked_fixes:
            entry = history.get(fix.fix)
            if entry and (entry["worked"] or entry["failed"] or entry["partial"]):
                fix.based_on_history = True
                fix.past_outcomes = past_outcomes_text(entry, service)
            else:
                fix.based_on_history = False
                fix.past_outcomes = "none"

        if memory_on and history:
            result.ranked_fixes = self._apply_memory_guardrail(result.ranked_fixes, history, service)

        if conflicts:
            line = ("Conflicting memory: " + ", ".join(conflicts)
                    + " worked at some point and failed later — treat these as uncertain.")
            if line not in result.evidence:
                result.evidence.append(line)
            if result.confidence == "high":
                result.confidence = "medium"

        if not result.evidence:
            result.evidence = ["Evidence was insufficient; the answer is best-effort."]
        return result

    @staticmethod
    def _apply_memory_guardrail(ranked: list[RankedFix],
                                history: dict[str, dict[str, Any]],
                                service: str = "") -> list[RankedFix]:
        """Stable-demote fixes with recorded failures; surface a missing known-good fix.

        Same-service outcomes outrank family-wide tallies: a fix that failed for
        this service is demoted even when it worked elsewhere, and one that worked
        here is surfaced even when the family-wide count is unimpressive.
        """

        def counts(fix: str) -> dict[str, Any] | None:
            entry = history.get(fix)
            if not entry:
                return None
            mine = (entry.get("by_service") or {}).get(service)
            if mine and (mine["worked"] or mine["failed"] or mine["partial"]):
                return mine
            return entry

        def bucket(fix: str) -> int:
            current = counts(fix)
            if current is None:
                return 1
            if current["worked"] and not current["failed"]:
                return 0
            if current["worked"]:
                return 1
            return 2 if current["failed"] else 1

        ordered = sorted(ranked, key=lambda f: bucket(f.fix))
        best = sorted(
            (fix for fix, e in history.items() if (counts(fix) or {}).get("worked", 0) > 0),
            key=lambda fix: ((counts(fix) or {}).get("worked", 0) - (counts(fix) or {}).get("failed", 0),
                             (counts(fix) or {}).get("worked", 0)),
            reverse=True,
        )
        if best and all(f.fix != best[0] for f in ordered):
            entry = history[best[0]]
            chosen = counts(best[0]) or entry
            where = f" on {service}" if chosen is not entry else " across services"
            ordered.insert(0, RankedFix(
                fix=best[0],
                rationale=("Added from memory: this fix has recorded successes for the same failure "
                           f"pattern ({chosen['worked']} worked{where})."),
                based_on_history=True,
                past_outcomes=past_outcomes_text(entry, service),
            ))
        return ordered[:5]

    # ---------------------------------------------------------------- agentic
    def _agentic(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                 memories: list[MemoryHit], memory_enabled: bool,
                 tracker: StepTracker
                 ) -> tuple[AgentResult, str, int, str | None, list[MemoryHit], dict[str, int]]:
        """v2: the model picks tools, capped at 6 calls, always falling back to v1."""
        started = time.perf_counter()
        collected: dict[str, Any] = {}
        used_memories = list(memories)
        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT + (
                    "\nYou may call at most 6 tools before answering. Prefer logs first, then deploys, "
                    "then metrics, then memory."
                )},
                {"role": "user", "content": (
                    f"Incident {inc.id} on {inc.service}: {inc.alert}. Symptoms: "
                    + "; ".join(inc.symptoms) + ". Investigate with the tools, then answer."
                )},
            ]
            for _ in range(6):
                message = self.llm.chat_tools(messages, tool_descriptions())
                if not message.tool_calls:
                    break
                for call in message.tool_calls:
                    name = call["name"]
                    args = call.get("arguments") or {}
                    if name == "get_logs":
                        value = self.world.get_logs(inc.service, inc.id,
                                                    int(args.get("last_n_lines", 45)))
                        detail = f"{value['line_count']} lines, {value['error_lines']} errors"
                    elif name == "get_recent_deploys":
                        value = self.world.get_recent_deploys(inc.service,
                                                              float(args.get("hours", 6)),
                                                              as_of=inc.opened_at)
                        detail = f"{len(value)} deploy(s)"
                    elif name == "get_metrics":
                        value = self.world.get_metrics(inc.service, inc.id)
                        detail = "snapshot loaded"
                    elif name == "recall_memory":
                        if self.memory.degraded:
                            raise MemoryUnavailable("Hindsight unavailable during agentic recall")
                        value = [m.model_dump() for m in self.memory.recall_similar(
                            alert=f"{inc.alert} {args.get('focus', '')}".strip(),
                            service=inc.service, symptoms=inc.symptoms,
                            logs_excerpt=self.world.log_excerpt(inc.id), k=5,
                            as_of=inc.opened_at)]
                        detail = f"{len(value)} memories"
                    else:
                        raise LLMError(f"model requested unknown tool {name}")
                    collected[name] = value
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "name": name,
                        "content": json.dumps(value)[:6000],
                    })
                    tracker.add(f"tool_{name}", f"Agent chose: {name}", detail=detail)

            logs = collected.get("get_logs", logs)
            deploys = collected.get("get_recent_deploys", deploys)
            metrics = collected.get("get_metrics", metrics)
            if "recall_memory" in collected:
                used_memories = [MemoryHit.model_validate(m) for m in collected["recall_memory"]]

            result, model, latency, fallback_reason, tokens = self._reason(
                inc, logs, deploys, metrics, used_memories, memory_enabled, tracker
            )
            return result, model, latency, fallback_reason, used_memories, tokens
        except Exception as exc:  # noqa: BLE001 - any agentic failure falls back
            tracker.add("agentic", "Agentic tool selection", status="error",
                        detail=f"{type(exc).__name__}: {exc}")
            result, model, latency, fallback_reason, tokens = self._reason(
                inc, logs, deploys, metrics, memories, memory_enabled, tracker
            )
            reason = f"agentic mode failed ({exc}); fell back to the deterministic pipeline"
            return result, model, latency, reason, memories, tokens


# --------------------------------------------------------------------------
# Outcome recording (shared by the UI, CLI and benchmark)
# --------------------------------------------------------------------------
def record_outcome(memory: MemoryBackend, incident: Incident | EvalIncident | dict,
                   fix: str, outcome: str, note: str = "") -> tuple[OutcomeRecord, str]:
    inc = incident if isinstance(incident, Incident) else Incident.model_validate(incident)
    record = OutcomeRecord(
        incident_id=inc.id,
        service=inc.service,
        family=inc.family,
        alert=inc.alert,
        fix=fix,
        outcome=outcome,  # type: ignore[arg-type]
        note=note,
        recorded_at=datetime.now(timezone.utc),
    )
    document_id = memory.retain_outcome(record)
    return record, document_id


def load_settings_and_incident(incident_id: str) -> tuple[Settings, dict]:
    from oncall.config import load_settings

    settings = load_settings()
    world = SimulatedWorld(settings.data_dir)
    incident = world.incident(incident_id)
    if incident is None:
        raise SystemExit(f"unknown incident {incident_id}")
    return settings, incident
