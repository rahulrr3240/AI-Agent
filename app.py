"""OnCall Memory — Streamlit UI.

Run: streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from oncall.agent import InvestigationAgent, record_outcome
from oncall.config import load_settings
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

    with st.spinner("Investigating…" if live else None):
        investigation = agent.investigate(incident_id, memory_enabled=memory_on, emit=emit)
    cache[key] = investigation
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

    st.markdown(
        f"{confidence_chip(result.confidence)}"
        f"{chip('memory ON' if investigation.memory_enabled else 'memory OFF')}"
        f"{chip(investigation.memory_backend)}",
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
    world = get_world()
    incident = world.incident(incident_id)
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

tab_demo, tab_learning = st.tabs(["🚨 Live incident", "📈 Learning curve"])

with tab_demo:
    left, center, right = st.columns([1.0, 2.15, 1.32], gap="large")

    # ------------------------------------------------------------- left feed
    with left:
        st.subheader("Incident feed")
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
        '<div class="oc-footer">Simulated data. Advisory only, no actions are executed. '
        'The agent’s learning is persistent memory — not model retraining.</div>',
        unsafe_allow_html=True,
    )

with tab_learning:
    st.subheader("Learning curve — measured, not drawn")
    st.caption(
        "Each eval incident is run three ways: memory OFF, memory ON with the seeded history, "
        "and memory ON where each simulated resolution is written back to memory. "
        "The chart is generated by `python -m oncall.evaluate`; the data is simulated."
    )
    reports = settings.reports_dir
    csv_path = reports / "benchmark_latest.csv"
    chart_path = reports / "learning_curve.png"

    if not csv_path.exists():
        st.info("No benchmark run yet. Run:\n\n"
                "```\npython -m oncall.evaluate --reset\n```")
    else:
        frame = pd.read_csv(csv_path)
        required = {"condition", "top1_correct", "attempts_to_resolve", "simulated_minutes"}
        if not required <= set(frame.columns):
            st.error("Benchmark CSV is missing expected columns.")
        else:
            summary = (frame.groupby("condition")
                       .agg(incidents=("incident_id", "count"),
                            top1_accuracy=("top1_correct", "mean"),
                            avg_attempts=("attempts_to_resolve", "mean"),
                            avg_minutes=("simulated_minutes", "mean"))
                       .reset_index())
            summary["condition"] = summary["condition"].map(
                lambda c: CONDITION_LABELS.get(c, c))
            display = summary.copy()
            display["top1_accuracy"] = (display["top1_accuracy"] * 100).round(1)
            display["avg_attempts"] = display["avg_attempts"].round(2)
            display["avg_minutes"] = display["avg_minutes"].round(1)
            st.dataframe(
                display.rename(columns={
                    "condition": "Condition", "incidents": "Incidents",
                    "top1_accuracy": "Top-1 accuracy (%)", "avg_attempts": "Avg attempts",
                    "avg_minutes": "Avg simulated minutes",
                }),
                hide_index=True, use_container_width=True,
            )
            if chart_path.exists():
                st.image(str(chart_path), caption="Cumulative top-1 accuracy (left) and "
                                                  "attempts per incident (right)")
            else:
                st.warning("Chart image not found; re-run the benchmark to regenerate it.")

            for column in ("llm_backend", "memory_backend"):
                if column in frame.columns:
                    values = sorted(frame[column].dropna().unique())
                    st.caption(f"{column.replace('_', ' ').title()}: {', '.join(map(str, values))}")
            if "llm_backend" in frame.columns and frame["llm_backend"].str.contains(
                    "heuristic", case=False, na=False).any():
                st.warning("This run used the deterministic fallback because no Groq key was "
                           "configured. Numbers measure the pipeline and memory, not the LLM.")
            with st.expander("Per-incident results"):
                st.dataframe(frame, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="oc-footer">Simulated data. Advisory only, no actions are executed.</div>',
        unsafe_allow_html=True,
    )
