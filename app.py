"""OnCall Memory — Streamlit UI.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from oncall.agent import InvestigationAgent, record_outcome
from oncall.config import load_settings
from oncall.llm import GroqLLM, LLMError
from oncall.live import (
    QUEUE_PATH,
    enqueue_manual,
    evidence_from_record,
    get_record,
    incident_from_record,
    load_queue,
    mark_investigated,
    parse_webhook,
)
from oncall.memory import (
    HindsightMemory,
    LocalMemory,
    MemoryUnavailable,
    build_memory,
)
from oncall.schemas import FIX_LABELS, Investigation, StepEvent
from oncall.tools import SimulatedWorld

APP_TITLE = "OnCall Memory"
TAGLINE = "Incident triage that remembers what actually fixed it."
CONDITION_LABELS = {
    "A_off": "A · Memory OFF (control)",
    "B_seeded": "B · Memory ON (seeded history)",
    "C_learning": "C · Memory ON + learning",
}
# The incidents DEMO_SCRIPT.md uses. Pre-running them means the recording never
# stalls on an API call, and compare mode reads the same cache.
DEMO_INCIDENTS = ["INC-103", "INC-101"]

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1560px; }
      html, body, [class*="st-"] { font-size: 16.5px; }
      h1 { font-size: 2.25rem !important; letter-spacing: -0.02em; }
      h2 { font-size: 1.45rem !important; }
      h3 { font-size: 1.15rem !important; }
      .oc-chip {
        display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
        border: 1px solid rgba(128,128,128,0.35); margin-right: 0.35rem;
      }
      .oc-alert { font-size: 1.28rem; font-weight: 650; line-height: 1.35; margin: 0.35rem 0; }
      .oc-cause { font-size: 1.18rem; line-height: 1.5; margin: 0.2rem 0 0.6rem 0; }
      .oc-mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.82rem; }
      .oc-muted { opacity: 0.72; font-size: 0.86rem; }
      .oc-footer { opacity: 0.65; font-size: 0.82rem; border-top: 1px solid rgba(128,128,128,0.25);
                   padding-top: 0.6rem; margin-top: 2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_settings():
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_world() -> SimulatedWorld:
    return SimulatedWorld(get_settings().data_dir)


@st.cache_resource(show_spinner=False)
def get_memory(bank_id: str):
    settings = get_settings()
    try:
        return build_memory(settings, bank_id=bank_id)
    except MemoryUnavailable as exc:
        # MEMORY_BACKEND=hindsight was explicit, but the bank is down: keep the
        # demo usable and make the degradation impossible to miss.
        fallback = LocalMemory(settings.base_dir / ".memory_store", bank_id)
        return fallback, f"Hindsight unavailable ({exc}). Using local keyword fallback."


@st.cache_resource(show_spinner=False)
def get_agent(bank_id: str) -> InvestigationAgent:
    settings = get_settings()
    memory, warning = get_memory(bank_id)
    return InvestigationAgent(settings, world=get_world(), memory=memory,
                              memory_warning=warning)


class _WarmProbe(BaseModel):
    ok: bool


@st.cache_resource(show_spinner="Warming up: one recall + one model call…")
def warm_up() -> dict[str, Any]:
    """Prove Hindsight and Groq work before the demo starts, not during it."""
    settings = get_settings()
    report: dict[str, Any] = {"memory_ok": False, "memory": "", "llm_ok": False, "llm": ""}
    try:
        memory, warning = get_memory(settings.hindsight_bank_id)
        if memory.name != "hindsight" or warning:
            report["memory"] = warning or "Hindsight unavailable; local keyword fallback is active"
        else:
            hits = memory.recall_similar(
                alert="checkout-service p95 latency > 2000ms for 5m",
                service="checkout-service", symptoms=["ConnectionPoolTimeoutError"],
                logs_excerpt="ConnectionPoolTimeoutError: Timeout waiting for connection from pool",
                k=1)
            report["memory_ok"] = True
            report["memory"] = (f"Hindsight OK — recall returned {len(hits)} memory"
                                if hits else
                                "Hindsight reachable but recall returned nothing — seed the bank")
    except MemoryUnavailable as exc:
        report["memory"] = f"Hindsight failed: {exc}"

    llm = GroqLLM(settings)
    if not llm.configured:
        report["llm"] = "Groq: GROQ_API_KEY is not set — the deterministic analyst will answer"
    else:
        try:
            result = llm.chat_json(
                [{"role": "system", "content": "Answer only with JSON."},
                 {"role": "user", "content": 'Return {"ok": true} exactly.'}],
                _WarmProbe, max_tokens=60)
            report["llm_ok"] = True
            report["llm"] = f"Groq OK — {result.model} in {result.latency_ms}ms"
        except LLMError as exc:
            report["llm"] = f"Groq failed: {str(exc)[:160]}"
    return report


def prerun_demo_incidents() -> None:
    """Run the scripted incidents now so the recording never waits on an API call.

    Results land in the same session cache the UI and compare mode read from, so
    the demo replays instantly afterwards.
    """
    progress = st.progress(0.0, text="Pre-running the demo incidents…")
    total = len(DEMO_INCIDENTS) * 2
    done = 0
    failures = []
    for incident_id in DEMO_INCIDENTS:
        for memory_on in (False, True):
            try:
                run_investigation(incident_id, memory_on)
            except Exception as exc:  # noqa: BLE001 - report, keep warming the rest
                failures.append(f"{incident_id} memory {'ON' if memory_on else 'OFF'}: "
                                f"{type(exc).__name__}: {exc}")
            done += 1
            progress.progress(done / total, text=f"Pre-running {incident_id} "
                                                 f"({'ON' if memory_on else 'OFF'}) — "
                                                 f"{done}/{total}")
    progress.empty()
    if failures:
        st.error("Some demo runs failed:\n\n" + "\n".join(f"- {f}" for f in failures))
    else:
        cached = ", ".join(f"{i} (OFF+ON)" for i in DEMO_INCIDENTS)
        st.success(f"Cached for the demo: {cached}. Compare mode reuses these runs.")


def chip(text: str, bg: str = "rgba(120,120,120,0.16)", fg: str = "inherit") -> str:
    return f'<span class="oc-chip" style="background:{bg};color:{fg}">{text}</span>'


def confidence_chip(confidence: str) -> str:
    colors = {
        "high": ("rgba(34,197,94,0.18)", "#15803d"),
        "medium": ("rgba(234,179,8,0.20)", "#a16207"),
        "low": ("rgba(239,68,68,0.18)", "#b91c1c"),
    }
    bg, fg = colors.get(confidence, colors["low"])
    return chip(f"confidence: {confidence}", bg, fg)


def severity_chip(severity: str) -> str:
    colors = {"P1": ("rgba(239,68,68,0.20)", "#b91c1c"),
              "P2": ("rgba(234,179,8,0.20)", "#a16207"),
              "P3": ("rgba(120,120,120,0.18)", "inherit")}
    bg, fg = colors.get(severity, colors["P3"])
    return chip(severity, bg, fg)


def step_icon(status: str) -> str:
    return {"done": "✅", "running": "⏳", "error": "⛔", "skipped": "⏭️"}.get(status, "•")


def render_trace(steps: list[StepEvent], placeholder=None) -> None:
    lines = []
    for step in steps:
        detail = f" — {step.detail}" if step.detail else ""
        lines.append(f"{step_icon(step.status)} **{step.label}**{detail}")
    text = "\n\n".join(lines) if lines else "_No steps yet._"
    if placeholder is not None:
        placeholder.markdown(text)
    else:
        st.markdown(text)


def resolve_incident(incident_id: str):
    """A queued live alert, or a record from the simulated world."""
    record = get_record(incident_id)
    if record is not None:
        return incident_from_record(record)
    return get_world().incident(incident_id)


def run_investigation(incident_id: str, memory_on: bool, live: bool = False) -> Investigation:
    cache: dict = st.session_state.setdefault("investigations", {})
    key = (incident_id, memory_on)
    if key in cache:
        return cache[key]

    agent = get_agent(get_settings().hindsight_bank_id)
    steps: list[StepEvent] = []
    placeholder = st.empty() if live else None
    if live:
        with st.container(border=True):
            st.caption("INVESTIGATION TRACE · live")
            placeholder = st.empty()
            placeholder.markdown("_Starting investigation…_")

    def emit(step: StepEvent) -> None:
        if step.status == "running":
            return
        steps[:] = [s for s in steps if s.name != step.name] + [step]
        if placeholder is not None:
            render_trace(steps, placeholder)

    record = get_record(incident_id)
    with st.spinner("Investigating…" if live else None):
        if record is not None:
            # Real input: same pipeline, evidence from what was supplied.
            investigation = agent.investigate(incident_from_record(record),
                                              memory_enabled=memory_on, emit=emit,
                                              evidence=evidence_from_record(record))
        else:
            investigation = agent.investigate(incident_id, memory_enabled=memory_on, emit=emit)
    cache[key] = investigation
    if record is not None:
        mark_investigated(incident_id)
    if placeholder is not None:
        render_trace(investigation.steps, placeholder)
    return investigation


# --------------------------------------------------------------------------
# Rendering pieces
# --------------------------------------------------------------------------
def render_result(investigation: Investigation, key_prefix: str,
                  show_outcomes: bool = True, compact: bool = False) -> None:
    result = investigation.result
    if investigation.memory_enabled and result.no_relevant_history:
        st.warning("**No relevant history found.** Recalled memories were from a different "
                   "failure pattern, so this is a clearly-labelled best-effort answer.")
    if not investigation.memory_enabled:
        st.info("**Memory is OFF.** No history was consulted — this is the control run.")

    if investigation.used_deterministic_fallback:
        engine = chip("Deterministic analyst (LLM unavailable)",
                      "rgba(239,68,68,0.20)", "#b91c1c")
    else:
        engine = chip(f"LLM · {investigation.llm_model}",
                      "rgba(34,197,94,0.18)", "#15803d")
    st.markdown(
        f"{confidence_chip(result.confidence)}"
        f"{chip('memory ON' if investigation.memory_enabled else 'memory OFF')}"
        f"{chip(investigation.memory_backend)}"
        f"{engine}",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="oc-cause"><b>Likely cause.</b> {result.likely_cause}</div>',
                unsafe_allow_html=True)

    st.markdown("**Evidence**")
    for line in result.evidence:
        st.markdown(f"- {line}")

    if result.similar_incidents:
        cited = ", ".join(f"**{s.id}**" for s in result.similar_incidents)
        st.markdown(f"**Similar past incidents:** {cited}")
        for s in result.similar_incidents:
            st.caption(f"{s.id}: {s.why_similar}")

    st.markdown("**Ranked fixes**")
    for index, fix in enumerate(result.ranked_fixes, start=1):
        label = FIX_LABELS.get(fix.fix, fix.fix)
        with st.container(border=True):
            header = f"**{index}. {label}**  ·  `{fix.fix}`"
            st.markdown(header)
            badges = []
            if fix.based_on_history:
                badges.append(chip("from memory", "rgba(59,130,246,0.18)", "#1d4ed8"))
                badges.append(chip(f"past: {fix.past_outcomes}"))
            else:
                badges.append(chip("no history"))
            st.markdown(" ".join(badges), unsafe_allow_html=True)
            st.write(fix.rationale)

            if show_outcomes and not compact:
                recorded = st.session_state.get("outcomes", {}).get(
                    f"{investigation.incident_id}|{fix.fix}")
                cols = st.columns([1, 1, 1, 3])
                note_key = f"{key_prefix}-note-{fix.fix}"
                clicked = None
                for col, outcome in zip(cols[:3], ("worked", "failed", "partial")):
                    if col.button(outcome.capitalize(), key=f"{key_prefix}-{outcome}-{fix.fix}",
                                  use_container_width=True):
                        clicked = outcome
                cols[3].text_input("Note (optional)", key=note_key, label_visibility="collapsed",
                                   placeholder="optional note, e.g. latency back to 380ms")
                if clicked:
                    handle_outcome(investigation.incident_id, fix.fix, clicked, note_key)
                if recorded:
                    st.caption(f"Recorded outcome: **{recorded}** (clicking again replaces it).")

    if investigation.used_deterministic_fallback:
        st.caption(f"Deterministic fallback answer — {investigation.fallback_reason}")
    elif investigation.llm_model:
        st.caption(f"Reasoned by {investigation.llm_model} in {investigation.llm_latency_ms}ms.")


def handle_outcome(incident_id: str, fix: str, outcome: str, note_key: str) -> None:
    settings = get_settings()
    memory, _ = get_memory(settings.hindsight_bank_id)
    incident = resolve_incident(incident_id)
    if incident is None:
        st.error(f"Cannot record an outcome: unknown incident {incident_id}.")
        return
    note = st.session_state.get(note_key, "")
    try:
        record, _document = record_outcome(memory, incident, fix, outcome, note)
    except MemoryUnavailable as exc:
        st.error(f"Could not store the outcome: {exc}")
        return
    st.session_state.setdefault("outcomes", {})[f"{incident_id}|{fix}"] = outcome
    st.session_state.pop("stats", None)
    st.session_state.pop("reflect", None)
    st.toast(f"Outcome stored: {record.fix} → {record.outcome}", icon="🧠")
    st.rerun()


def render_compare(investigation: Investigation, key_prefix: str) -> None:
    result = investigation.result
    st.markdown(
        f"#### {'🧠 Memory ON' if investigation.memory_enabled else '🚫 Memory OFF'}"
    )
    if investigation.memory_enabled and result.no_relevant_history:
        st.warning("No relevant history found.")
    st.markdown(f"{confidence_chip(result.confidence)}", unsafe_allow_html=True)
    st.markdown(f"**Why:** {result.likely_cause}")
    st.markdown("**Top fixes**")
    for index, fix in enumerate(result.ranked_fixes[:3], start=1):
        badge = (f" · *{fix.past_outcomes}*" if fix.based_on_history else "")
        st.markdown(f"{index}. **{FIX_LABELS.get(fix.fix, fix.fix)}** (`{fix.fix}`){badge}")
    st.caption(f"{len(investigation.memories)} memories used · {investigation.llm_model}")
    with st.expander("Evidence"):
        for line in result.evidence:
            st.markdown(f"- {line}")
    if investigation.memories:
        with st.expander("Memories used"):
            for memory in investigation.memories:
                st.markdown(f"**{memory.metadata.get('incident_id', 'memory')}** — "
                            f"{memory.text[:260]}…")


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
settings = get_settings()
world = get_world()
memory, memory_warning = get_memory(settings.hindsight_bank_id)
incidents = world.evals()
incident_ids = [incident["id"] for incident in incidents]

if "selected_id" not in st.session_state:
    st.session_state.selected_id = incident_ids[0]
st.session_state.setdefault("memory_on", True)
st.session_state.setdefault("stats", None)

header_left, header_right = st.columns([3, 1.2])
with header_left:
    st.title(APP_TITLE)
    st.caption(TAGLINE)
with header_right:
    backend_label = "Hindsight" if memory.name == "hindsight" else "Local keyword fallback"
    st.markdown(chip(f"{backend_label} · bank {settings.hindsight_bank_id}",
                     "rgba(59,130,246,0.16)", "#1d4ed8"), unsafe_allow_html=True)
    if not settings.has_groq:
        st.markdown(chip("Groq key missing · deterministic analyst"), unsafe_allow_html=True)

if memory_warning:
    st.warning(memory_warning)
if not settings.has_hindsight:
    st.info("To use real memory, set `HINDSIGHT_BASE_URL` and `HINDSIGHT_API_KEY` in `.env`, "
            "then run `python scripts/seed_memory.py --force`.")

warm = warm_up()
if warm["memory_ok"] and warm["llm_ok"]:
    st.success(f"● Warm-up passed — {warm['memory']} · {warm['llm']}")
else:
    st.error(f"● Warm-up FAILED — {warm['memory']} · {warm['llm']}")

with st.expander("Demo prep", expanded=False):
    st.caption(
        f"DEMO_SCRIPT.md walks through {', '.join(DEMO_INCIDENTS)}. Pre-run them once before "
        "you record: each run is cached in this session, so the live trace and the "
        "OFF-vs-ON comparison replay instantly instead of waiting on the API."
    )
    if st.button("Pre-run the scripted demo incidents", use_container_width=True):
        prerun_demo_incidents()
    cached = sorted({f"{i} ({'ON' if on else 'OFF'})"
                     for (i, on) in st.session_state.get("investigations", {})})
    if cached:
        st.caption("Cached in this session: " + ", ".join(cached))

tab_demo, tab_live, tab_learning = st.tabs(
    ["🚨 Live incident", "🔌 Live alert (real input)", "📈 Learning curve"])

with tab_demo:
    left, center, right = st.columns([1.0, 2.15, 1.32], gap="large")

    # ------------------------------------------------------------- left feed
    with left:
        st.subheader("Incident feed")
        queued_live = load_queue(limit=5)
        if queued_live:
            st.markdown("**⚡ Queued live alerts**")
            for record in queued_live:
                st.caption(f"`{record['id']}` · {record['service']} · "
                           f"{record['alert'][:64]}")
            st.caption("Real input, not fixtures — open the **Live alert** tab to investigate one.")
            st.divider()
        if st.button("Trigger next incident ▶", use_container_width=True, type="primary"):
            current = incident_ids.index(st.session_state.selected_id)
            st.session_state.selected_id = incident_ids[(current + 1) % len(incident_ids)]
            st.session_state.pop("compare_id", None)
            st.rerun()
        st.selectbox("Or pick one", incident_ids, key="selected_id",
                     format_func=lambda i: f"{i} · {world.incident(i)['service']}")

        selected = world.incident(st.session_state.selected_id)
        opened = datetime.fromisoformat(selected["opened_at"].replace("Z", "+00:00"))
        with st.container(border=True):
            st.markdown(
                f"{severity_chip(selected.get('severity', 'P2'))}"
                f"{chip(selected['service'])}"
                f"{chip(opened.strftime('%Y-%m-%d %H:%M UTC'))}",
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="oc-alert">{selected["alert"]}</div>', unsafe_allow_html=True)
            st.caption("Symptoms")
            for symptom in selected.get("symptoms", []):
                st.markdown(f"- {symptom}")
            deploy_id = selected.get("related_deploy_id")
            if deploy_id:
                deploy = next((d for d in world.deploys() if d["id"] == deploy_id), None)
                if deploy:
                    st.caption(f"Most recent relevant deploy: {deploy['service']} "
                               f"{deploy['version']} at {deploy['deployed_at']}")
        st.caption(f"{len(incidents)} unresolved eval incidents · data is simulated")

    # ---------------------------------------------------------- center agent
    with center:
        st.subheader("Agent investigation")
        try:
            current_inv = run_investigation(st.session_state.selected_id,
                                            st.session_state.memory_on, live=True)
        except Exception as exc:  # noqa: BLE001 - surface, never crash the demo
            st.error(f"Investigation failed: {type(exc).__name__}: {exc}")
            current_inv = None
        if current_inv:
            st.markdown("#### Result")
            render_result(current_inv, key_prefix=f"{current_inv.incident_id}-main")

    # --------------------------------------------------------- right memory
    with right:
        st.subheader("Memory")
        st.toggle("Memory ON", key="memory_on",
                  help="OFF skips recall entirely — the control condition.")
        if memory.name != "hindsight":
            st.caption("Degraded mode: local keyword retrieval. Not Hindsight.")
        else:
            st.caption(f"Connected to {settings.hindsight_base_url}")

        if st.button("Run memory OFF vs ON comparison", use_container_width=True,
                     type="secondary", disabled=not st.session_state.memory_on):
            st.session_state.compare_id = st.session_state.selected_id
            st.rerun()

        if st.session_state.get("stats") is None:
            try:
                st.session_state.stats = memory.stats()
            except MemoryUnavailable as exc:
                st.session_state.stats = {"error": str(exc)}
        stats = st.session_state.stats
        if isinstance(stats, dict) and "error" in stats:
            st.error(f"Memory stats unavailable: {stats['error']}")
        elif stats:
            cols = st.columns(2)
            cols[0].metric("Incidents", stats.get("incidents", 0))
            cols[1].metric("Outcomes", stats.get("outcomes", 0))
            if st.button("Refresh memory stats", use_container_width=True):
                st.session_state.pop("stats", None)
                st.rerun()
            if stats.get("reliable_fixes"):
                st.caption("Most reliable recorded fixes")
                for row in stats["reliable_fixes"][:3]:
                    st.markdown(f"- `{row['fix']}` · {row['service']} · "
                                f"{row['worked']} worked / {row['failed']} failed")

        st.divider()
        st.markdown("**Memories used for this answer**")
        if current_inv is None:
            st.caption("No investigation yet.")
        elif not current_inv.memory_enabled:
            st.caption("Memory is OFF — nothing was recalled.")
        elif not current_inv.memories:
            st.caption("No relevant memories were found for this incident.")
        else:
            for memory_hit in current_inv.memories:
                incident_id = memory_hit.metadata.get("incident_id", memory_hit.id)
                record_type = memory_hit.metadata.get("record_type", "incident")
                with st.expander(f"{incident_id} · {record_type}"):
                    st.markdown(memory_hit.text)
                    if memory_hit.metadata.get("outcomes"):
                        st.caption("Recorded: " + memory_hit.metadata["outcomes"].replace(";", " · "))

        st.divider()
        st.markdown("**What we know about this service**")
        service = selected["service"]
        if st.button(f"Summarise {service}", use_container_width=True):
            with st.spinner("Reflecting over memory…"):
                try:
                    summary = memory.reflect_pattern(service)
                    st.session_state.setdefault("reflect", {})[service] = summary
                except MemoryUnavailable as exc:
                    st.error(f"Reflect failed: {exc}")
        summary = st.session_state.get("reflect", {}).get(service)
        if summary:
            with st.container(border=True):
                st.markdown(summary)

    # ----------------------------------------------------------- compare mode
    compare_id = st.session_state.get("compare_id")
    if compare_id:
        st.divider()
        st.subheader(f"Compare mode · {compare_id}")
        st.caption("The same alert, same model, same logs. Only the recall step differs.")
        compare_off = run_investigation(compare_id, False)
        compare_on = run_investigation(compare_id, True)
        col_off, col_on = st.columns(2, gap="large")
        with col_off:
            render_compare(compare_off, f"{compare_id}-off")
        with col_on:
            render_compare(compare_on, f"{compare_id}-on")

    st.markdown(
        '<div class="oc-footer">Seeded history and the incident fixtures are simulated; live '
        'alerts are real input you supplied. Advisory only — no action is executed. The '
        'agent’s learning is persistent memory — not model retraining.</div>',
        unsafe_allow_html=True,
    )

with tab_live:
    st.subheader("Investigate an alert you actually received")
    st.caption(
        "This tab takes real input — service, alert text, a webhook body, your own logs. "
        "The pipeline is the same one the simulated incidents use: recall from Hindsight, "
        "reasoning on Groq, and Worked / Failed / Partial written back as memory. Anything "
        "you do not supply is reported as missing instead of invented."
    )
    live_left, live_center = st.columns([1.02, 1.98], gap="large")

    with live_left:
        st.markdown(f"**Alert queue** · `.{QUEUE_PATH.parent.name}/{QUEUE_PATH.name}`")
        queued = load_queue(limit=12)
        if not queued:
            st.caption("Empty. Queue one below, or POST a real notification:")
            st.code("python webhook.py\npython scripts/send_alert.py", language="bash")
        for record in queued:
            received = record.get("received_at", "")[:16].replace("T", " ")
            with st.container(border=True):
                st.markdown(
                    f"{severity_chip(record.get('severity', 'P2'))}"
                    f"{chip(record['service'])}"
                    f"{chip(record['family'])}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{record['alert'][:150]}**")
                st.caption(f"`{record['id']}` · {received} UTC · from {record['source']} · "
                           f"{'logs pasted' if record.get('logs_text') else 'no logs'} · "
                           f"{record.get('status', 'queued')}")
                if st.button("Investigate", key=f"live-open-{record['id']}",
                             use_container_width=True):
                    st.session_state.live_selected_id = record["id"]
                    st.session_state.live_run = record["id"]

        st.divider()
        st.markdown("#### Queue an alert by hand")
        service_options = sorted(world.services())
        with st.form("live_alert_form"):
            service_choice = st.selectbox("Service", service_options + ["other (type below)…"],
                                          help="List comes from data/services.json; type any name.")
            service_other = st.text_input("Service name", key="live_service_other",
                                          placeholder="billing-gateway")
            alert_text = st.text_area(
                "Alert text", height=90,
                placeholder="checkout-service p95 latency > 2000ms for 5m; "
                            "ConnectionPoolTimeoutError in logs")
            webhook_body = st.text_area(
                "Alertmanager / Grafana webhook JSON (optional)", height=150,
                placeholder='{"alerts":[{"labels":{"service":"checkout-service"},'
                            '"annotations":{"description":"…"}}]}')
            log_paste = st.text_area("Log extract (paste what you have)", height=160)
            log_upload = st.file_uploader("…or upload a log file", type=["log", "txt"],
                                          key="live_log_file")
            form_submit = st.form_submit_button("Queue and investigate", type="primary",
                                                use_container_width=True)

        if form_submit:
            payload = None
            if webhook_body.strip():
                try:
                    payload = json.loads(webhook_body)
                except json.JSONDecodeError as exc:
                    st.error(f"Webhook body is not valid JSON: {exc}")
                    payload = None
            service = (service_other.strip() or "unknown-service"
                       if service_choice.startswith("other") else service_choice)
            if payload and not service_choice.startswith("other"):
                webhook_alerts = parse_webhook(payload)
                if webhook_alerts and webhook_alerts[0].get("service"):
                    service = ""
            if not alert_text.strip() and not payload:
                st.error("Give me the alert text, or paste an Alertmanager/Grafana body.")
            else:
                pasted = log_paste
                if log_upload is not None:
                    pasted = (pasted + "\n"
                              + log_upload.getvalue().decode("utf-8", errors="replace")).strip()
                record = enqueue_manual(service, alert_text.strip(), pasted, world.families(),
                                        source="manual", origin="typed into the app",
                                        webhook=payload if isinstance(payload, dict) else None)
                st.session_state.live_selected_id = record["id"]
                st.session_state.live_run = record["id"]
                st.rerun()

    with live_center:
        live_id = st.session_state.get("live_selected_id")
        record = get_record(live_id) if live_id else None
        if record is None and queued:
            record = queued[0]
        if record is None:
            st.info("Nothing to investigate yet. Queue an alert on the left — the same "
                    "recall, reasoning and feedback loop runs on it as on the fixtures.")
        else:
            st.markdown(
                f"{severity_chip(record.get('severity', 'P2'))}"
                f"{chip(record['service'])}"
                f"{chip(record['family'])}"
                f"{chip('live input · ' + record['source'])}",
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="oc-alert">{record["alert"]}</div>', unsafe_allow_html=True)
            supplied, missing = [], []
            supplied.append(f"`{record['id']}`")
            (supplied if record.get("logs_text") else missing).append("logs")
            (supplied if record.get("deploys") else missing).append("deploys")
            (supplied if record.get("metrics") else missing).append("metrics")
            st.caption("Supplied: " + (", ".join(supplied) or "—")
                       + (" · Not supplied: " + ", ".join(missing) if missing else ""))
            live_memory_on = st.toggle("Memory ON", key="live_memory_on", value=True,
                                       help="OFF recalls nothing — the control path.")
            if st.session_state.get("live_run") != record["id"]:
                st.caption("Press investigate to recall history and reason over this alert.")
                if st.button("🔍 Investigate this live alert", key=f"live-run-{record['id']}",
                             type="primary", use_container_width=True):
                    st.session_state.live_run = record["id"]
            if st.session_state.get("live_run") == record["id"]:
                try:
                    live_inv = run_investigation(record["id"], live_memory_on, live=True)
                except Exception as exc:  # noqa: BLE001 - surface, never crash the demo
                    st.error(f"Investigation failed: {type(exc).__name__}: {exc}")
                    live_inv = None
                if live_inv:
                    st.markdown("#### Result")
                    render_result(live_inv, key_prefix=f"{record['id']}-live")
                    st.caption("Worked / Failed / Partial retains the outcome in Hindsight under "
                               "this live incident's ID, so the next alert of this pattern "
                               "starts from it.")

    st.markdown(
        '<div class="oc-footer">Seeded history and the incident fixtures are simulated. '
        'Anything on this tab is real input you supplied. Advisory only: no action is ever '
        'executed, and the agent\'s learning is persistent memory — not model retraining.</div>',
        unsafe_allow_html=True,
    )

with tab_learning:
    st.subheader("Learning curve — measured, not drawn")
    st.caption(
        "Results are produced by `python -m oncall.evaluate`; they are shown in this "
        "Learning curve tab and in `reports/benchmark_<run_id>.csv`. This README states "
        "no benchmark numbers."
    )
    reports = settings.reports_dir
    json_path = reports / "benchmark_latest.json"
    csv_path = reports / "benchmark_latest.csv"
    run_csv_path = csv_path
    meta: dict[str, Any] = {}
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        run_id = str(meta.get("run_id") or "")
        if run_id:
            candidate = reports / f"benchmark_{run_id}.csv"
            if candidate.exists():
                run_csv_path = candidate

    if not run_csv_path.exists():
        st.warning(
            "**The full A/B/C benchmark could not be completed.** No benchmark CSV was found "
            "in the reports directory, so there are no validated numbers to compare."
        )
        st.info("No benchmark run yet. Run:\n\n"
                "```\npython -m oncall.evaluate --reset\n```")
    else:
        frame = pd.read_csv(run_csv_path)
        required = {"condition", "top1_correct", "attempts_to_resolve",
                    "simulated_minutes", "llm_answered"}
        if not required <= set(frame.columns):
            st.error("Benchmark CSV is missing expected columns.")
        else:
            status = str(meta.get("status") or "incomplete")
            run_id = str(meta.get("run_id") or run_csv_path.stem.replace("benchmark_", ""))
            st.write(f"Run ID: {run_id}")
            st.write(f"Status: {status}")

            rows: list[dict[str, Any]] = []
            for condition in sorted(frame["condition"].dropna().unique()):
                sub = frame[frame["condition"] == condition].copy()
                answered = sub[sub["llm_answered"].fillna(False).astype(bool)].copy()
                rows.append({
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "total_rows": int(len(sub)),
                    "llm_answered_rows": int(len(answered)),
                    "top1": float(answered["top1_correct"].mean()) if not answered.empty else None,
                    "resolved": float(answered["resolved"].mean()) if not answered.empty else None,
                    "avg_attempts": float(answered["attempts_to_resolve"].mean()) if not answered.empty else None,
                    "avg_minutes": float(answered["simulated_minutes"].mean()) if not answered.empty else None,
                    "llm_failed_rows": int((sub["llm_backend"] == "llm_failed").sum())
                    if "llm_backend" in sub.columns else 0,
                })
            summary = pd.DataFrame(rows)

            if summary.empty:
                st.warning(
                    "**The full A/B/C benchmark could not be completed.** The benchmark file "
                    "contains no rows to compare."
                )
            else:
                display = summary.copy()
                display["LLM answered / total"] = display.apply(
                    lambda row: f"{int(row['llm_answered_rows'])} / {int(row['total_rows'])}", axis=1)
                display["Top-1"] = display.apply(
                    lambda row: (f"{row['top1'] * 100:.1f}% (n={int(row['llm_answered_rows'])})"
                                 if row["top1"] is not None else "n/a"), axis=1)
                display["Resolved"] = display.apply(
                    lambda row: (f"{row['resolved'] * 100:.1f}% (n={int(row['llm_answered_rows'])})"
                                 if row["resolved"] is not None else "n/a"), axis=1)
                display["Avg attempts"] = display.apply(
                    lambda row: (f"{row['avg_attempts']:.2f} (n={int(row['llm_answered_rows'])})"
                                 if row["avg_attempts"] is not None else "n/a"), axis=1)
                display["Avg simulated minutes"] = display.apply(
                    lambda row: (f"{row['avg_minutes']:.1f} (n={int(row['llm_answered_rows'])})"
                                 if row["avg_minutes"] is not None else "n/a"), axis=1)
                display["LLM failed rows"] = display["llm_failed_rows"].astype(int)

                st.dataframe(
                    display.rename(columns={
                        "condition_label": "Condition",
                        "LLM answered / total": "LLM answered / total",
                        "Top-1": "Top-1",
                        "Resolved": "Resolved",
                        "Avg attempts": "Avg attempts",
                        "Avg simulated minutes": "Avg simulated minutes",
                        "LLM failed rows": "LLM failed rows",
                    })[["Condition", "LLM answered / total", "Top-1", "Resolved",
                       "Avg attempts", "Avg simulated minutes", "LLM failed rows"]],
                    hide_index=True, use_container_width=True,
                )

                compared = summary[summary["llm_answered_rows"] > 0].copy()
                if compared.empty:
                    st.caption("too few LLM-answered rows to compare conditions; no improvement is shown by this run.")
                else:
                    if (compared["llm_answered_rows"] < 10).any():
                        st.caption("too few LLM-answered rows to compare conditions; no improvement is shown by this run.")
                    else:
                        text_bits = []
                        for _, row in compared.sort_values("top1", ascending=False).iterrows():
                            label = row["condition_label"]
                            top1 = row["top1"]
                            if top1 is None:
                                continue
                            text_bits.append(f"{label}: top-1 {top1 * 100:.1f}% (n={int(row['llm_answered_rows'])})")
                        if text_bits:
                            st.caption(
                                "Compared conditions show: " + ", ".join(text_bits)
                                + ". This is a neutral summary of the table above; no improvement "
                                "claim is made beyond the values shown."
                            )

                chart_path = reports / "learning_curve.png"
                plotted = summary[summary["llm_answered_rows"] > 0].copy()
                if chart_path.exists() and not plotted.empty and (plotted["llm_answered_rows"] >= 1).all():
                    st.image(str(chart_path), use_container_width=True)

                for column in ("llm_backend", "memory_backend"):
                    if column in frame.columns:
                        values = sorted(frame[column].dropna().unique())
                        st.caption(f"{column.replace('_', ' ').title()}: {', '.join(map(str, values))}")

                st.subheader("Cold start: does it improve as outcomes are recorded?")
                d_rows = frame[(frame["condition"] == "D_cold") &
                               (frame["llm_answered"].fillna(False).astype(bool))].sort_values("order")
                if d_rows.empty:
                    st.warning(
                        "**The full A/B/C benchmark could not be completed.** No D-condition rows "
                        "were answered by the model, so there is no cold-start learning signal to compare yet."
                    )
                else:
                    d_rows = d_rows.copy()
                    d_rows["cumulative_top1"] = (
                        d_rows["top1_correct"].cumsum() /
                        pd.Series(range(1, len(d_rows) + 1), index=d_rows.index)
                    ) * 100
                    st.line_chart(d_rows.set_index("order")["cumulative_top1"])
                    st.caption(
                        "Cumulative top-1 accuracy for condition D using only LLM-answered rows."
                    )

                if "llm_backend" in frame.columns and frame["llm_backend"].str.contains(
                        "heuristic", case=False, na=False).any():
                    st.warning(
                        "This run used the deterministic fallback because no Groq key was "
                        "configured. Numbers measure the pipeline and memory, not the LLM."
                    )

                with st.expander("Per-incident results"):
                    st.dataframe(frame, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="oc-footer">Benchmark incidents, deploys, logs and metrics are simulated. '
        'Live alerts (Live alert tab) are real input. Advisory only — no action is executed.</div>',
        unsafe_allow_html=True,
    )
