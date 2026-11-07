"""Learning-curve benchmark: memory OFF vs seeded memory vs memory with learning.

Three conditions, each on a fresh throwaway bank (never the demo bank):

  A_off       memory OFF — the same pipeline without recall (control)
  B_seeded    memory ON, the 28 resolved history incidents pre-seeded
  C_learning  memory ON, seeded, and every simulated fix outcome written back
              after each incident, so later incidents can benefit

Every number comes from real agent runs plus the simulated engineer; nothing is
hand-drawn. The CSV, JSON summary and chart land in reports/.

  python -m oncall.evaluate
  python -m oncall.evaluate --limit 8          # quick smoke run
  python -m oncall.evaluate --keep-bank        # inspect the banks afterwards
  python -m oncall.evaluate --reset --bank oncall-eval-manual
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oncall.agent import InvestigationAgent, record_outcome
from oncall.config import Settings, load_settings
from oncall.memory import MemoryBackend, MemoryUnavailable, build_memory
from oncall.simulator import SimulatedEngineer
from oncall.tools import SimulatedWorld

CONDITIONS = ["A_off", "B_seeded", "C_learning"]
CONDITION_LABELS = {
    "A_off": "A · Memory OFF",
    "B_seeded": "B · Memory ON (seeded)",
    "C_learning": "C · Memory ON + learning",
}
CONDITION_COLORS = {"A_off": "#9ca3af", "B_seeded": "#3b82f6", "C_learning": "#16a34a"}
CONDITION_STYLES = {"A_off": "-", "B_seeded": "--", "C_learning": "-"}
CONDITION_MARKERS = {"A_off": "o", "B_seeded": "s", "C_learning": "o"}


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


def run_condition(settings: Settings, world: SimulatedWorld, condition: str,
                  run_id: str, incidents: list[dict], agent_factory=None
                  ) -> tuple[list[dict[str, Any]], MemoryBackend, str | None]:
    bank = _condition_settings(settings, run_id, condition)
    memory, warning = build_memory(settings, bank_id=bank)
    print(f"  {condition}: backend={memory.name} bank={bank}", flush=True)
    if warning:
        print(f"    WARNING: {warning}", flush=True)

    if condition in ("B_seeded", "C_learning"):
        _seed(memory, world, condition)

    agent = (agent_factory or (lambda mem: InvestigationAgent(
        settings, world=world, memory=mem, memory_warning=warning)))(memory)
    rows: list[dict[str, Any]] = []
    for order, raw in enumerate(incidents, start=1):
        memory_on = condition != "A_off"
        investigation = agent.investigate(raw["id"], memory_enabled=memory_on)
        resolution = SimulatedEngineer().resolve(raw, investigation.result.ranked_fixes)

        if condition == "C_learning":
            for attempt in resolution.attempts:
                record_outcome(memory, raw, attempt.fix, attempt.outcome, attempt.note)

        flag_correct: bool | None = None
        if raw.get("novel") and memory_on:
            flag_correct = bool(investigation.result.no_relevant_history)

        rows.append({
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
            "llm_backend": ("heuristic (fallback)" if investigation.used_deterministic_fallback
                            else investigation.llm_model),
            "llm_model": investigation.llm_model,
            "fallback_reason": investigation.fallback_reason or "",
        })
        print(f"    {order:02d}/{len(incidents)} {raw['id']}: top={resolution.top_fix} "
              f"{'OK' if resolution.top1_correct else 'miss'} · "
              f"{resolution.attempts_to_resolve} attempt(s) · {resolution.simulated_minutes} min",
              flush=True)

    try:
        memory.close()
    except Exception:  # noqa: BLE001 - close is best effort
        pass
    return rows, memory, warning


def summarise(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        sub = frame[frame["condition"] == condition]
        if sub.empty:
            continue
        novel = sub[sub["novel"]]
        summary[condition] = {
            "incidents": int(len(sub)),
            "top1_accuracy": round(float(sub["top1_correct"].mean()), 4),
            "resolved_rate": round(float(sub["resolved"].mean()), 4),
            "avg_attempts": round(float(sub["attempts_to_resolve"].mean()), 3),
            "avg_simulated_minutes": round(float(sub["simulated_minutes"].mean()), 2),
            "second_half_top1": round(float(sub.iloc[len(sub) // 2:]["top1_correct"].mean()), 4),
            "novel_incidents": int(len(novel)),
            "novel_flag_accuracy": (
                round(float(novel["no_history_flag_correct"].dropna().mean()), 4)
                if not novel.empty else None
            ),
            "fallback_runs": int(sub["llm_backend"].str.contains("heuristic").sum()),
        }
    return summary


def make_chart(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=170)
    for condition in CONDITIONS:
        sub = frame[frame["condition"] == condition].sort_values("order")
        if sub.empty:
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

    axes[0].set_title("Cumulative top-1 accuracy", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("% incidents where the first-ranked fix was correct")
    axes[0].set_ylim(0, 105)
    axes[1].set_title("Attempts to resolve", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("attempts (dots) · cumulative average (line)")
    for axis in axes:
        axis.set_xlabel("eval incident number")
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("OnCall Memory — measured on 20 simulated eval incidents "
                 "(3 conditions, fresh memory banks)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}", flush=True)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="only the first N eval incidents")
    parser.add_argument("--conditions", default="A,B,C", help="subset of A,B,C")
    parser.add_argument("--bank", default=None, help="base bank id (default: unique per run)")
    parser.add_argument("--reset", action="store_true",
                        help="delete the three banks before running (use with --bank)")
    parser.add_argument("--keep-bank", action="store_true",
                        help="keep the throwaway banks (default: delete after the run)")
    parser.add_argument("--chart-only", action="store_true",
                        help="redraw the chart from the existing CSV (no API calls)")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.chart_only:
        csv_path = settings.reports_dir / "benchmark_latest.csv"
        chart_path = settings.reports_dir / "learning_curve.png"
        make_chart(pd.read_csv(csv_path), chart_path)
        print(f"Redrew {chart_path} from {csv_path}")
        return 0
    world = SimulatedWorld(settings.data_dir)
    incidents = world.evals()
    if args.limit:
        incidents = incidents[:args.limit]

    selected = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    wanted = []
    for token in selected:
        matches = [c for c in CONDITIONS if c.startswith(token)]
        if not matches:
            parser.error(f"unknown condition {token!r}; use A, B or C")
        wanted.append(matches[0])
    wanted = [c for c in CONDITIONS if c in wanted]

    run_id = args.bank or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    if run_id == settings.hindsight_bank_id:
        parser.error("refusing to run the benchmark against the demo bank")

    print(f"Benchmark run {run_id}")
    print(f"Backend: {'Hindsight' if settings.has_hindsight else 'local keyword fallback'}"
          f" | model: {settings.groq_model if settings.has_groq else 'deterministic heuristic'}")
    print(f"Incidents: {len(incidents)} | conditions: {', '.join(wanted)}")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    backends: dict[str, Any] = {}
    warnings: dict[str, str | None] = {}
    started = time.perf_counter()

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
        rows, memory, warning = run_condition(settings, world, condition, run_id, incidents)
        all_rows.extend(rows)
        backends[condition] = memory.name
        warnings[condition] = warning
        print(f"  {condition} finished in {time.perf_counter() - began:.1f}s", flush=True)

    frame = pd.DataFrame(all_rows)
    summary = summarise(frame)
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "incidents": len(incidents),
        "conditions": wanted,
        "hindsight_configured": settings.has_hindsight,
        "groq_configured": settings.has_groq,
        "models": {"primary": settings.groq_model, "fallback": settings.groq_fallback_model},
        "memory_backends": backends,
        "memory_warnings": warnings,
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    csv_path = settings.reports_dir / "benchmark_latest.csv"
    json_path = settings.reports_dir / "benchmark_latest.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.bank:
        frame.to_csv(settings.reports_dir / f"benchmark_{run_id}.csv", index=False)
        (settings.reports_dir / f"benchmark_{run_id}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    chart_path = settings.reports_dir / "learning_curve.png"
    make_chart(frame, chart_path)

    print("\nSummary")
    for condition, stats in summary.items():
        print(f"  {CONDITION_LABELS[condition]}: top-1 {stats['top1_accuracy']:.0%}, "
              f"avg attempts {stats['avg_attempts']:.2f}, "
              f"avg {stats['avg_simulated_minutes']:.1f} min, "
              f"resolved {stats['resolved_rate']:.0%}")
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
