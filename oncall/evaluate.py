"""Learning-curve benchmark: memory OFF vs seeded vs learning vs cold start.

Four conditions, each on a fresh throwaway bank (never the demo bank):

  A_off       memory OFF — the same pipeline without recall (control)
  B_seeded    memory ON, the 28 resolved history incidents pre-seeded
  C_learning  memory ON, seeded, and every simulated fix outcome written back
              after each incident, so later incidents can benefit
  D_cold      memory ON with an EMPTY bank: nothing is seeded, every simulated fix
              outcome is written back, so the run can only learn from itself

Every number comes from real agent runs plus the simulated engineer; nothing is
hand-drawn. Each row is written to the run's CSV as soon as that incident
finishes, so one run can be spread across several days.

  python -m oncall.evaluate
  python -m oncall.evaluate --limit 8          # quick smoke run
  python -m oncall.evaluate --no-fallback      # only LLM-answered rows count
  python -m oncall.evaluate --tpm 6500         # pace under the Groq free-tier TPM limit
  python -m oncall.evaluate --pace 20          # minimum seconds between runs
  python -m oncall.evaluate --keep-bank        # inspect the banks afterwards
  python -m oncall.evaluate --reset --bank oncall-eval-manual
  python -m oncall.evaluate --conditions A,B   # subset of A,B,C,D
  python -m oncall.evaluate --resume <run_id>  # continue a run from its own CSV
  python -m oncall.evaluate --resume <run_id> --retry-failed --conditions A,B # retry failed rows
  python -m oncall.evaluate --resume <run_id> --retry-failed  # re-run failed rows (A and B only)

With `--no-fallback` a Groq failure is recorded as `llm_failed` and excluded
from the headline numbers instead of being answered by the deterministic
heuristic. That matters most for condition A: the heuristic always ranks
`restart_service` first, so a heuristic row in A is not a model answer at all.
With `--retry-failed`, any incident whose latest row has `llm_backend == "llm_failed"`
is re-run and appended to the CSV; this is permitted for conditions A and B only.
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from oncall.agent import REASONING_MAX_TOKENS, InvestigationAgent, record_outcome
from oncall.config import Settings, load_settings
from oncall.llm import LLMError
from oncall.memory import MemoryBackend, MemoryUnavailable, build_memory
from oncall.simulator import SimulatedEngineer
from oncall.tools import SimulatedWorld

CONDITIONS = ["A_off", "B_seeded", "C_learning", "D_cold"]
CONDITION_LABELS = {
    "A_off": "A · Memory OFF",
    "B_seeded": "B · Memory ON (seeded)",
    "C_learning": "C · Memory ON + learning",
    "D_cold": "D · Memory ON (cold start)",
}
CONDITION_COLORS = {"A_off": "#9ca3af", "B_seeded": "#3b82f6", "C_learning": "#16a34a",
                    "D_cold": "#f59e0b"}
CONDITION_STYLES = {"A_off": "-", "B_seeded": "--", "C_learning": "-", "D_cold": ":"}
CONDITION_MARKERS = {"A_off": "o", "B_seeded": "s", "C_learning": "o", "D_cold": "^"}


class Pacer:
    """Sliding-window throttle for the Groq free tier (tokens per minute).

    It sleeps on *measured* usage: after each investigation it knows how many
    prompt + completion tokens that call really cost, keeps a 60-second window
    of them, and waits until the window fits under `tpm` again.
    """

    def __init__(self, tpm: int = 8000, minimum_seconds: float = 2.0) -> None:
        self.tpm = max(0, int(tpm))
        self.minimum = max(0.0, float(minimum_seconds))
        self.events: list[tuple[float, int]] = []
        self.waited_seconds = 0.0
        self.waits = 0

    def wait(self, tokens: int) -> None:
        now = time.monotonic()
        self.events = [(t, n) for t, n in self.events if now - t < 60.0]
        reservation = max(0, int(tokens))
        delay = self.minimum
        if self.tpm and reservation > self.tpm:
            delay = max(delay, 60.0)
        elif self.tpm:
            used = sum(n for _, n in self.events)
            drop = 0
            while used + reservation > self.tpm and drop < len(self.events):
                used -= self.events[drop][1]
                delay = max(delay, self.events[drop][0] + 60.0 - now + 0.5)
                drop += 1
        if delay > 0:
            self.waits += 1
            self.waited_seconds += delay
            print(f"    pacing: reserving {reservation} tokens against "
                  f"{sum(n for _, n in self.events)}/{self.tpm} tokens/60s "
                  f"→ waiting {delay:.1f}s", flush=True)
            time.sleep(delay)
        self.events.append((time.monotonic(), reservation))


def _condition_settings(settings: Settings, run_id: str, condition: str) -> str:
    return f"oncall-eval-{run_id}-{condition.lower()}"


def _seed(memory: MemoryBackend, world: SimulatedWorld, label: str) -> None:
    history = world.history()
    started = time.perf_counter()
    for index, raw in enumerate(history, start=1):
        memory.retain_incident(raw)
        if index % 7 == 0 or index == len(history):
            print(f"    [{label}] seeded {index}/{len(history)} incidents", flush=True)
    print(f"    [{label}] seed complete in {time.perf_counter() - started:.1f}s", flush=True)


ROW_COLUMNS = [
    "run_id", "condition", "order", "incident_id", "family", "service", "novel",
    "true_fix", "top_fix", "top1_correct", "attempts_to_resolve", "resolved",
    "simulated_minutes", "tried", "no_relevant_history", "no_history_flag_correct",
    "memories_used", "detected_pattern", "memory_backend", "llm_backend", "llm_model",
    "llm_answered", "prompt_tokens", "completion_tokens", "fallback_reason",
    "retry_of_failed",
]


def _run_csv_path(reports_dir: Path, run_id: str) -> Path:
    return reports_dir / f"benchmark_{run_id}.csv"


def _run_json_path(reports_dir: Path, run_id: str) -> Path:
    return reports_dir / f"benchmark_{run_id}.json"


def _append_row(path: Path, row: dict[str, Any]) -> None:
    """Persist one finished row immediately, so an interrupted run is resumable."""
    frame = pd.DataFrame([row]).reindex(columns=ROW_COLUMNS)
    fresh = not (path.exists() and path.stat().st_size > 0)
    frame.to_csv(path, mode="a", header=fresh, index=False)


def read_run_rows(path: Path) -> pd.DataFrame:
    """Every row recorded for a run so far (empty frame when there is none yet)."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=ROW_COLUMNS)
    frame = pd.read_csv(path).reindex(columns=ROW_COLUMNS)
    for column in ("llm_answered", "top1_correct", "resolved", "novel", "retry_of_failed"):
        frame[column] = (frame[column].astype(str).str.lower()
                         .isin(("true", "1")).astype(bool))
    return frame


def run_condition(settings: Settings, world: SimulatedWorld, condition: str,
                  run_id: str, incidents: list[dict], agent_factory=None,
                  pacer: Pacer | None = None, no_fallback: bool = False,
                  done_ids: set[str] | None = None,
                  retry_ids: set[str] | None = None,
                  on_row: Callable[[dict[str, Any]], None] | None = None
                  ) -> tuple[list[dict[str, Any]], MemoryBackend, str | None]:
    bank = _condition_settings(settings, run_id, condition)
    memory, warning = build_memory(settings, bank_id=bank)
    print(f"  {condition}: backend={memory.name} bank={bank}"
          + (" | fallback disabled" if no_fallback else ""), flush=True)
    if warning:
        print(f"    WARNING: {warning}", flush=True)

    if condition in ("B_seeded", "C_learning"):
        _seed(memory, world, condition)

    agent = (agent_factory or (lambda mem: InvestigationAgent(
        settings, world=world, memory=mem, memory_warning=warning,
        allow_fallback=not no_fallback)))(memory)
    rows: list[dict[str, Any]] = []
    last_tokens = 3000          # used to keep pacing honest when a call fails
    retry_set = retry_ids or set()

    def emit(row: dict[str, Any]) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    for order, raw in enumerate(incidents, start=1):
        if done_ids and raw["id"] in done_ids:
            print(f"    {order:02d}/{len(incidents)} {raw['id']}: already recorded — skipped",
                  flush=True)
            continue
        is_retry = raw["id"] in retry_set
        memory_on = condition != "A_off"
        if pacer:
            pacer.wait(last_tokens)
        try:
            investigation = agent.investigate(raw["id"], memory_enabled=memory_on)
        except LLMError as exc:
            # --no-fallback: this row is not an LLM answer, so it is recorded as
            # failed and kept out of the headline numbers.
            emit({
                "run_id": run_id, "condition": condition, "order": order,
                "incident_id": raw["id"], "family": raw["family"], "service": raw["service"],
                "novel": bool(raw.get("novel", False)), "true_fix": raw["true_fix"],
                "top_fix": "", "top1_correct": False, "attempts_to_resolve": 0,
                "resolved": False, "simulated_minutes": 0, "tried": "",
                "no_relevant_history": False, "no_history_flag_correct": None,
                "memories_used": 0, "detected_pattern": "", "memory_backend": memory.name,
                "llm_backend": "llm_failed", "llm_model": "none", "llm_answered": False,
                "prompt_tokens": 0, "completion_tokens": 0,
                "fallback_reason": str(exc)[:300],
                "retry_of_failed": is_retry,
            })
            print(f"    {order:02d}/{len(incidents)} {raw['id']}: LLM FAILED — row excluded "
                  f"from headline numbers", flush=True)
            continue

        resolution = SimulatedEngineer().resolve(raw, investigation.result.ranked_fixes)

        if condition in ("C_learning", "D_cold"):
            for attempt in resolution.attempts:
                record_outcome(memory, raw, attempt.fix, attempt.outcome, attempt.note)

        flag_correct: bool | None = None
        if raw.get("novel") and memory_on:
            flag_correct = bool(investigation.result.no_relevant_history)

        answered = not investigation.used_deterministic_fallback
        # Groq meters `prompt + max_tokens` against its TPM limit, so pacing uses
        # the reservation the call made, not just what it ended up generating.
        reserved = (investigation.llm_prompt_tokens + REASONING_MAX_TOKENS
                    if answered and investigation.llm_prompt_tokens else last_tokens)
        last_tokens = reserved
        emit({
            "run_id": run_id,
            "condition": condition,
            "order": order,
            "incident_id": raw["id"],
            "family": raw["family"],
            "service": raw["service"],
            "novel": bool(raw.get("novel", False)),
            "true_fix": raw["true_fix"],
            "top_fix": resolution.top_fix,
            "top1_correct": resolution.top1_correct,
            "attempts_to_resolve": resolution.attempts_to_resolve,
            "resolved": resolution.resolved,
            "simulated_minutes": resolution.simulated_minutes,
            "tried": ";".join(resolution.tried_fixes),
            "no_relevant_history": investigation.result.no_relevant_history,
            "no_history_flag_correct": flag_correct,
            "memories_used": len(investigation.memories),
            "detected_pattern": investigation.detected_pattern,
            "memory_backend": investigation.memory_backend,
            "llm_backend": ("heuristic (fallback)" if not answered
                            else investigation.llm_model),
            "llm_model": investigation.llm_model,
            "llm_answered": answered,
            "prompt_tokens": investigation.llm_prompt_tokens,
            "completion_tokens": investigation.llm_completion_tokens,
            "fallback_reason": investigation.fallback_reason or "",
            "retry_of_failed": is_retry,
        })
        print(f"    {order:02d}/{len(incidents)} {raw['id']}: top={resolution.top_fix} "
              f"{'OK' if resolution.top1_correct else 'miss'} · "
              f"{resolution.attempts_to_resolve} attempt(s) · {resolution.simulated_minutes} min"
              f" · {last_tokens} tok reserved"
              + ("" if answered else " · HEURISTIC FALLBACK"),
              flush=True)

    try:
        memory.close()
    except Exception:  # noqa: BLE001 - close is best effort
        pass
    return rows, memory, warning


def summarise(frame: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers use only rows the LLM actually answered.

    Heuristic-fallback and llm_failed rows stay in the CSV and are counted here,
    but they never move a headline percentage: the deterministic analyst ranks
    `restart_service` first whenever memory is off, so counting those rows in
    condition A would be measuring a strawman.
    """
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        sub = frame[frame["condition"] == condition]
        if sub.empty:
            continue
        latest_sub = sub.drop_duplicates(subset=["incident_id"], keep="last")
        superseded = sub.loc[~sub.index.isin(latest_sub.index)]
        retried_count = int((superseded["llm_backend"] == "llm_failed").sum())
        answered = (latest_sub[latest_sub["llm_answered"].astype(bool)]
                    if "llm_answered" in latest_sub else latest_sub)
        novel = answered[answered["novel"]]
        has_rows = len(answered) > 0
        summary[condition] = {
            "incidents": int(len(latest_sub)),
            "llm_answered": int(len(answered)),
            "fallback_rows": int(latest_sub["llm_backend"].str.contains("heuristic").sum()),
            "llm_failed_rows": int((latest_sub["llm_backend"] == "llm_failed").sum()),
            "retried_rows": retried_count,
            "headline_basis": ("rows answered by the LLM only"
                               if len(answered) != len(latest_sub) else "all rows were LLM-answered"),
            "top1_accuracy": round(float(answered["top1_correct"].mean()), 4) if has_rows else None,
            "resolved_rate": round(float(answered["resolved"].mean()), 4) if has_rows else None,
            "avg_attempts": round(float(answered["attempts_to_resolve"].mean()), 3) if has_rows else None,
            "avg_simulated_minutes": round(float(answered["simulated_minutes"].mean()), 2) if has_rows else None,
            "second_half_top1": round(float(answered.iloc[len(answered) // 2:]["top1_correct"].mean()), 4)
            if has_rows else None,
            "novel_incidents": int(len(novel)),
            "novel_flag_accuracy": (
                round(float(novel["no_history_flag_correct"].dropna().mean()), 4)
                if not novel.empty else None
            ),
            "avg_prompt_tokens": round(float(answered["prompt_tokens"].mean()), 1)
            if has_rows and "prompt_tokens" in answered else None,
            "all_rows_top1": round(float(latest_sub["top1_correct"].mean()), 4),
        }
    return summary


def make_chart(frame: pd.DataFrame, path: Path, *, incomplete: bool = False) -> None:
    """Cumulative top-1 per condition, LLM-answered rows only.

    A condition without LLM-answered rows is not drawn (an empty or single-point
    line would read as a result); it is named in the title instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frame.empty and "condition" in frame.columns and "incident_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["condition", "incident_id"], keep="last")

    total = int(frame["incident_id"].nunique()) if "incident_id" in frame.columns else 0
    if "llm_answered" in frame.columns:
        frame = frame[frame["llm_answered"].astype(bool)]

    plotted: list[str] = []
    without_rows: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=170)
    for condition in CONDITIONS:
        sub = frame[frame["condition"] == condition].sort_values("order")
        if sub.empty:
            without_rows.append(CONDITION_LABELS[condition])
            continue
        label = CONDITION_LABELS[condition]
        color = CONDITION_COLORS[condition]
        style = CONDITION_STYLES[condition]
        marker = CONDITION_MARKERS[condition]
        cumulative = (sub["top1_correct"].cumsum()
                      / pd.Series(range(1, len(sub) + 1), index=sub.index))
        axes[0].plot(sub["order"], cumulative * 100, marker=marker, linewidth=2.2,
                     linestyle=style, markersize=5, color=color, label=label)
        axes[1].scatter(sub["order"], sub["attempts_to_resolve"], color=color,
                        alpha=0.22, s=32)
        axes[1].plot(sub["order"], sub["attempts_to_resolve"].expanding().mean(),
                     marker=marker, linewidth=2.2, linestyle=style, markersize=5,
                     color=color, label=label)
        plotted.append(label)

    notes: list[str] = []
    if incomplete:
        notes.append("INCOMPLETE RUN")
    if without_rows:
        notes.append("no LLM-answered rows: " + ", ".join(without_rows))
    if len(plotted) == 1:
        notes.append("only one condition had LLM-answered rows")
    suffix = (" · " + " · ".join(notes)) if notes else ""

    if plotted:
        axes[0].set_title("Cumulative top-1 accuracy", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("% incidents where the first-ranked fix was correct")
        axes[0].set_ylim(0, 105)
        axes[1].set_title("Attempts to resolve", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("attempts (dots) · cumulative average (line)")
        for axis in axes:
            axis.set_xlabel("eval incident number")
            axis.grid(alpha=0.25, linewidth=0.7)
            axis.legend(frameon=False, fontsize=9)
    else:
        for axis in axes:
            axis.axis("off")
            axis.text(0.5, 0.5, "No LLM-answered rows\nin this run",
                      ha="center", va="center", fontsize=13, transform=axis.transAxes)

    fig.suptitle(f"OnCall Memory — measured on {total} simulated eval incidents "
                 f"(fresh memory banks, LLM-answered rows only){suffix}",
                 fontsize=11 if notes else 13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}", flush=True)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="only the first N eval incidents")
    parser.add_argument("--conditions", default="A,B,C,D", help="subset of A,B,C,D")
    parser.add_argument("--bank", default=None, help="base bank id (default: unique per run)")
    parser.add_argument("--resume", default=None, metavar="RUN_ID",
                        help="continue an existing run from reports/benchmark_<run_id>.csv, "
                             "skipping incidents already recorded for each condition")
    parser.add_argument("--retry-failed", action="store_true",
                        help="with --resume, re-run incidents whose latest row failed (A and B only)")
    parser.add_argument("--reset", action="store_true",
                        help="delete the selected banks before running (use with --bank)")
    parser.add_argument("--keep-bank", action="store_true",
                        help="keep the throwaway banks (default: delete after the run)")
    parser.add_argument("--chart-only", action="store_true",
                        help="redraw the chart from the existing CSV (no API calls)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="record a row as llm_failed instead of using the deterministic "
                             "heuristic when Groq fails; those rows are excluded from the "
                             "headline numbers")
    parser.add_argument("--tpm", type=int, default=6500,
                        help="token budget per minute to pace against (Groq free tier caps "
                             "openai/gpt-oss-120b at 8000 TPM; the default leaves headroom "
                             "because repair retries and completion output reserve more than "
                             "they use; 0 disables pacing)")
    parser.add_argument("--pace", type=float, default=2.0,
                        help="minimum seconds to wait between investigations")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.chart_only:
        csv_path = settings.reports_dir / "benchmark_latest.csv"
        json_path = settings.reports_dir / "benchmark_latest.json"
        chart_path = settings.reports_dir / "learning_curve.png"
        status = ""
        if json_path.exists():
            try:
                status = str(json.loads(json_path.read_text(encoding="utf-8"))
                             .get("status", ""))
            except (json.JSONDecodeError, OSError):
                status = ""
        make_chart(read_run_rows(csv_path), chart_path, incomplete=status == "incomplete")
        print(f"Redrew {chart_path} from {csv_path}")
        return 0
    world = SimulatedWorld(settings.data_dir)
    all_incidents = world.evals(include_truth=True)

    selected = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    wanted = []
    for token in selected:
        matches = [c for c in CONDITIONS if c.startswith(token)]
        if not matches:
            parser.error(f"unknown condition {token!r}; use A, B, C or D")
        wanted.append(matches[0])
    wanted = [c for c in CONDITIONS if c in wanted]

    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if args.retry_failed:
        forbidden = [c for c in wanted if c in ("C_learning", "D_cold")]
        if forbidden:
            parser.error(
                "retrying would change the memory state seen by later incidents; rerun the condition instead"
            )

    run_id = (args.resume or args.bank
              or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}")
    if run_id == settings.hindsight_bank_id:
        parser.error("refusing to run the benchmark against the demo bank")
    if args.no_fallback and not settings.has_groq:
        parser.error("--no-fallback needs a GROQ_API_KEY: without it every row would be llm_failed")

    run_csv = _run_csv_path(settings.reports_dir, run_id)
    run_json = _run_json_path(settings.reports_dir, run_id)
    prior = read_run_rows(run_csv)
    prior_meta: dict[str, Any] = {}
    if args.resume:
        if prior.empty and not run_json.exists():
            parser.error(f"nothing to resume for run {run_id!r}: {run_csv.name} is missing")
        if run_json.exists():
            try:
                prior_meta = json.loads(run_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                prior_meta = {}

    # On resume the incident set comes from the run's own record, so a later
    # invocation cannot silently shrink or reorder the run.
    recorded_ids = [str(i) for i in (prior_meta.get("incident_ids") or [])]
    if args.resume and recorded_ids:
        by_id = {raw["id"]: raw for raw in all_incidents}
        absent = [i for i in recorded_ids if i not in by_id]
        if absent:
            parser.error(f"resume would lose incidents missing from data/: {absent[:5]}")
        incidents = [by_id[i] for i in recorded_ids]
    else:
        incidents = all_incidents[:args.limit] if args.limit else all_incidents
    incident_ids = [raw["id"] for raw in incidents]

    done: dict[str, set[str]] = {}
    retries: dict[str, set[str]] = {}
    for condition in wanted:
        if prior.empty:
            done[condition] = set()
            retries[condition] = set()
            continue
        cond_prior = prior[prior["condition"] == condition]
        if cond_prior.empty:
            done[condition] = set()
            retries[condition] = set()
            continue
        latest_cond = cond_prior.drop_duplicates(subset=["incident_id"], keep="last")
        if args.retry_failed:
            failed_mask = latest_cond["llm_backend"] == "llm_failed"
            done[condition] = set(latest_cond.loc[~failed_mask, "incident_id"])
            retries[condition] = set(latest_cond.loc[failed_mask, "incident_id"])
        else:
            done[condition] = set(latest_cond["incident_id"])
            retries[condition] = set()

    pacer = Pacer(tpm=args.tpm, minimum_seconds=args.pace)
    print(f"Benchmark run {run_id}")
    print(f"Backend: {'Hindsight' if settings.has_hindsight else 'local keyword fallback'}"
          f" | model: {settings.groq_model if settings.has_groq else 'deterministic heuristic'}")
    print(f"Incidents: {len(incidents)} | conditions: {', '.join(wanted)}")
    print(f"Pacing: {args.tpm} tokens/minute, at least {args.pace:g}s between investigations "
          f"| fallback: {'disabled (llm_failed rows)' if args.no_fallback else 'enabled'}")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    backends: dict[str, Any] = {}
    warnings: dict[str, str | None] = {}
    started = time.perf_counter()

    if not prior.empty:
        print(f"Resuming {run_id}: {len(prior)} row(s) already recorded")

    def sink(row: dict[str, Any]) -> None:
        _append_row(run_csv, row)

    for condition in wanted:
        bank = _condition_settings(settings, run_id, condition)
        if args.reset:
            existing, _ = build_memory(settings, bank_id=bank)
            try:
                existing.delete_bank()
                print(f"  reset: deleted {bank}")
            except MemoryUnavailable as exc:
                print(f"  reset skipped for {bank}: {exc}")
        began = time.perf_counter()
        rows, memory, warning = run_condition(settings, world, condition, run_id, incidents,
                                              pacer=pacer, no_fallback=args.no_fallback,
                                              done_ids=done.get(condition, set()),
                                              retry_ids=retries.get(condition, set()),
                                              on_row=sink)
        backends[condition] = memory.name
        warnings[condition] = warning
        print(f"  {condition} finished in {time.perf_counter() - began:.1f}s "
              f"({len(rows)} new row(s))", flush=True)

    # Summary and chart always come from every row in the run's own CSV, so a
    # run split across several invocations is still one run.
    frame = read_run_rows(run_csv)
    summary = summarise(frame)
    present = set(frame["condition"]) if not frame.empty else set()
    intended = [c for c in CONDITIONS if c in present or c in wanted]
    missing_rows = {
        condition: [i for i in incident_ids
                    if i not in set(frame.loc[frame["condition"] == condition, "incident_id"])]
        for condition in intended
    }
    missing_rows = {c: ids for c, ids in missing_rows.items() if ids}
    complete = bool(intended) and not missing_rows

    total_retried = sum(int(s.get("retried_rows") or 0) for s in summary.values())
    total_failed = sum(int(s.get("llm_failed_rows") or 0) for s in summary.values())

    payload = {
        "run_id": run_id,
        "status": "complete" if complete else "incomplete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "incidents": len(incident_ids),
        "incident_ids": incident_ids,
        "conditions": intended,
        "conditions_this_invocation": wanted,
        "missing_rows": missing_rows,
        "rows_already_recorded": int(len(prior)),
        "retried_rows": total_retried,
        "remaining_llm_failed": total_failed,
        "hindsight_configured": settings.has_hindsight,
        "groq_configured": settings.has_groq,
        "models": {"primary": settings.groq_model, "fallback": settings.groq_fallback_model},
        "memory_backends": backends,
        "memory_warnings": {c: warnings.get(c) for c in wanted if warnings.get(c)},
        "method": {
            "headline_basis": "top-1/resolved/attempts/minutes are computed on rows the LLM "
                              "answered; heuristic-fallback and llm_failed rows are counted "
                              "and excluded",
            "resumable": True,
            "retry_failed": bool(args.retry_failed),
            "no_fallback": bool(args.no_fallback),
            "tokens_per_minute_budget": int(args.tpm),
            "minimum_pace_seconds": float(args.pace),
            "seconds_spent_pacing": round(pacer.waited_seconds, 1),
            "pacing_waits": pacer.waits,
            "prompt_log_chars": 3000,
        },
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    csv_path = settings.reports_dir / "benchmark_latest.csv"
    json_path = settings.reports_dir / "benchmark_latest.json"
    frame.to_csv(csv_path, index=False)
    body = json.dumps(payload, indent=2) + "\n"
    json_path.write_text(body, encoding="utf-8")
    run_json.write_text(body, encoding="utf-8")

    chart_path = settings.reports_dir / "learning_curve.png"
    make_chart(frame, chart_path, incomplete=not complete)

    print(f"\nRun {run_id}: {payload['status'].upper()}"
          + (f" — missing rows in {', '.join(sorted(missing_rows))}" if missing_rows else ""))
    for condition, stats in summary.items():
        def pct(value: Any) -> str:
            return "n/a" if value is None else f"{value:.0%}"

        def num(value: Any, places: int = 2) -> str:
            return "n/a" if value is None else f"{value:.{places}f}"

        print(f"  {CONDITION_LABELS[condition]}: top-1 {pct(stats['top1_accuracy'])} "
              f"of {stats['llm_answered']}/{stats['incidents']} LLM-answered rows · "
              f"resolved {pct(stats['resolved_rate'])} · attempts "
              f"{num(stats['avg_attempts'])} · minutes {num(stats['avg_simulated_minutes'], 1)}")
        if stats["fallback_rows"] or stats["llm_failed_rows"] or stats.get("retried_rows"):
            parts: list[str] = []
            if stats["fallback_rows"] or stats["llm_failed_rows"]:
                parts.append(f"{stats['fallback_rows']} heuristic fallback(s)")
                parts.append(f"{stats['llm_failed_rows']} llm_failed")
            if stats.get("retried_rows"):
                parts.append(f"{stats['retried_rows']} llm_failed rows retried")
            print(f"      excluded: {', '.join(parts)} · all-rows top-1 would be "
                  f"{pct(stats['all_rows_top1'])}")
    print(f"  retries: {total_retried} rows retried, {total_failed} remain llm_failed")
    print(f"  pacing: {pacer.waits} wait(s), {pacer.waited_seconds:.0f}s of the "
          f"{payload['elapsed_seconds']:.0f}s run")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {chart_path}")

    if not args.keep_bank:
        for condition in wanted:
            bank = _condition_settings(settings, run_id, condition)
            try:
                temp, _ = build_memory(settings, bank_id=bank)
                temp.delete_bank()
            except MemoryUnavailable as exc:
                print(f"  cleanup failed for {bank}: {exc}")
        print("Throwaway banks deleted (use --keep-bank to inspect them).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
