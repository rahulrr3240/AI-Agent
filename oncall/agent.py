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
from oncall.llm import GroqLLM, LLMError
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

SYSTEM_PROMPT = """You are OnCall Memory, an incident-response copilot for a production engineering team.

Rules:
1. Use ONLY the supplied incident context (logs, deploys, metrics, recalled memories). Do not invent services, versions, log lines or incident IDs.
2. Cite specific evidence: exact log lines, deploy versions/times or metric values.
3. Memory is advisory. Prefer fixes with recorded successes. A fix recorded as FAILED for the same failure pattern must be demoted below fixes that worked, and you must say why.
4. If a fix worked once and failed later, mention the conflict and lower confidence.
5. If the recalled memories are not genuinely similar to this incident, set no_relevant_history=true and give a clearly generic best-effort answer with confidence "low". Do not cite any incident IDs in that case.
6. Only cite incident IDs that appear in the recalled memories. Put the reason each one is similar in why_similar.
7. ranked_fixes use ONLY ids from the supplied FIX VOCABULARY, ordered best first.
8. The answer is advisory: no action is executed.

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


class InvestigationAgent:
    def __init__(self, settings: Settings, world: SimulatedWorld | None = None,
                 memory: MemoryBackend | None = None, llm: GroqLLM | None = None,
                 memory_warning: str | None = None) -> None:
        self.settings = settings
        self.world = world or SimulatedWorld(settings.data_dir)
        if memory is None:
            memory, detected_warning = build_memory(settings)
            memory_warning = memory_warning or detected_warning
        self.memory = memory
        self.memory_warning = memory_warning
        self.llm = llm or GroqLLM(settings)

    # ------------------------------------------------------------------ core
    def investigate(self, incident: Incident | EvalIncident | dict,
                    memory_enabled: bool = True,
                    emit: Callable[[StepEvent], None] | None = None) -> Investigation:
        if isinstance(incident, str):
            found = self.world.incident(incident)
            if found is None:
                raise ValueError(f"unknown incident {incident}")
            incident = found
        inc = incident if isinstance(incident, Incident) else Incident.model_validate(incident)
        tracker = StepTracker(emit)
        warning = self.memory_warning
        effective_memory = memory_enabled

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
                        logs_excerpt=self.world.log_excerpt(inc.id), k=8, as_of=inc.opened_at,
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
            result, model, latency, fallback_reason, memories = self._agentic(
                inc, logs, deploys, metrics, memories, memory_enabled, tracker
            )
        else:
            result, model, latency, fallback_reason = self._reason(
                inc, logs, deploys, metrics, memories, effective_memory, tracker
            )

        history = fix_history_from_memories(memories) if effective_memory else {}
        conflicts = conflicting_fixes(history) if effective_memory else []
        result = self._postprocess(result, memories, history, effective_memory, conflicts)

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
            used_deterministic_fallback=bool(fallback_reason),
            fallback_reason=fallback_reason,
            conflicts=conflicts,
        )

    # --------------------------------------------------------------- reasoning
    def _reason(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                memories: list[MemoryHit], memory_on: bool,
                tracker: StepTracker) -> tuple[AgentResult, str, int, str | None]:
        started = time.perf_counter()
        tracker.add("reasoning", "Reasoning over evidence", status="running")

        if not self.llm.configured:
            result = self._heuristic(inc, logs, deploys, metrics, memories, memory_on)
            duration = int((time.perf_counter() - started) * 1000)
            tracker.add("reasoning", "Reasoning over evidence",
                        status="done",
                        detail="Groq key not configured — deterministic heuristic analyst",
                        duration_ms=duration)
            return result, "heuristic (no GROQ_API_KEY)", duration, "GROQ_API_KEY is not set"

        messages = self._build_messages(inc, logs, deploys, metrics, memories, memory_on)
        try:
            llm_result = self.llm.chat_json(messages, AgentResult, max_tokens=1800)
        except LLMError as exc:
            result = self._heuristic(inc, logs, deploys, metrics, memories, memory_on)
            duration = int((time.perf_counter() - started) * 1000)
            tracker.add("reasoning", "Reasoning over evidence", status="done",
                        detail="Groq failed after retries — deterministic heuristic used",
                        duration_ms=duration)
            return result, "heuristic (Groq failed)", duration, str(exc)

        duration = int((time.perf_counter() - started) * 1000)
        detail = f"{llm_result.model} · {llm_result.latency_ms}ms · {llm_result.attempts} attempt(s)"
        if llm_result.notes:
            detail += " · " + "; ".join(llm_result.notes)
        tracker.add("reasoning", "Reasoning over evidence", status="done", detail=detail,
                    duration_ms=duration)
        return llm_result.data, llm_result.model, llm_result.latency_ms, None

    def _build_messages(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                        memories: list[MemoryHit], memory_on: bool) -> list[dict[str, str]]:
        history = fix_history_from_memories(memories) if memory_on else {}
        history_rows = [
            {
                "fix": fix,
                "worked": entry["worked"],
                "failed": entry["failed"],
                "partial": entry["partial"],
                "incidents": sorted(entry["incidents"]),
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
            "logs": logs.get("text", "")[-6000:],
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
            + json.dumps(context, indent=2)
            + "\n\nSchema (all fields required):\n"
            + json.dumps(AgentResult.model_json_schema(), indent=2)[:5000]
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
            summary = metrics["summary"]
            evidence.append(
                f"Metrics: peak p95 {summary['peak_p95_ms']}ms, "
                f"peak error rate {summary['peak_error_rate_pct']}%"
            )

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
            ordered = sorted(
                history.items(),
                key=lambda kv: (kv[1]["worked"] > 0, kv[1]["worked"] - kv[1]["failed"], kv[1]["worked"]),
                reverse=True,
            )
            for fix, entry in ordered[:4]:
                if entry["worked"] == 0:
                    continue
                ranked.append(RankedFix(
                    fix=fix,
                    rationale=(f"Recorded {entry['worked']} success(es) and {entry['failed']} failure(s) "
                               f"for this pattern in past incidents."),
                    based_on_history=True,
                    past_outcomes=past_outcomes_text(entry),
                ))
        for fix in _generic_fixes():
            if len(ranked) >= 3:
                break
            if all(r.fix != fix for r in ranked):
                ranked.append(RankedFix(
                    fix=fix,
                    rationale="Generic best-effort step; no matching history was consulted.",
                    based_on_history=False,
                    past_outcomes=past_outcomes_text(history.get(fix)),
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
                     conflicts: list[str]) -> AgentResult:
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
                fix.past_outcomes = past_outcomes_text(entry)
            else:
                fix.based_on_history = False
                fix.past_outcomes = "none"

        if memory_on and history:
            result.ranked_fixes = self._apply_memory_guardrail(result.ranked_fixes, history)

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
                                history: dict[str, dict[str, Any]]) -> list[RankedFix]:
        """Stable-demote fixes with recorded failures; surface a missing known-good fix."""

        def bucket(fix: str) -> int:
            entry = history.get(fix)
            if not entry:
                return 1
            if entry["worked"] and not entry["failed"]:
                return 0
            if entry["worked"]:
                return 1
            return 2 if entry["failed"] else 1

        ordered = sorted(ranked, key=lambda f: bucket(f.fix))
        best = sorted(
            (fix for fix, e in history.items() if e["worked"] > 0),
            key=lambda fix: (history[fix]["worked"] - history[fix]["failed"], history[fix]["worked"]),
            reverse=True,
        )
        if best and all(f.fix != best[0] for f in ordered):
            entry = history[best[0]]
            ordered.insert(0, RankedFix(
                fix=best[0],
                rationale=("Added from memory: this fix has recorded successes for the same failure "
                           f"pattern ({entry['worked']} worked)."),
                based_on_history=True,
                past_outcomes=past_outcomes_text(entry),
            ))
        return ordered[:5]

    # ---------------------------------------------------------------- agentic
    def _agentic(self, inc: Incident, logs: dict, deploys: list[dict], metrics: dict,
                 memories: list[MemoryHit], memory_enabled: bool,
                 tracker: StepTracker
                 ) -> tuple[AgentResult, str, int, str | None, list[MemoryHit]]:
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

            result, model, latency, fallback_reason = self._reason(
                inc, logs, deploys, metrics, used_memories, memory_enabled, tracker
            )
            return result, model, latency, fallback_reason, used_memories
        except Exception as exc:  # noqa: BLE001 - any agentic failure falls back
            tracker.add("agentic", "Agentic tool selection", status="error",
                        detail=f"{type(exc).__name__}: {exc}")
            result, model, latency, fallback_reason = self._reason(
                inc, logs, deploys, metrics, memories, memory_enabled, tracker
            )
            reason = f"agentic mode failed ({exc}); fell back to the deterministic pipeline"
            return result, model, latency, reason, memories


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
