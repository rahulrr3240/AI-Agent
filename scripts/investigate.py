"""Run one incident through the investigation pipeline from the terminal.

  python scripts/investigate.py INC-103
  python scripts/investigate.py INC-103 --memory off
  python scripts/investigate.py INC-103 --compare
  python scripts/investigate.py INC-103 --json > result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oncall.agent import InvestigationAgent  # noqa: E402
from oncall.config import load_settings  # noqa: E402
from oncall.memory import build_memory  # noqa: E402
from oncall.schemas import Investigation  # noqa: E402
from oncall.tools import SimulatedWorld  # noqa: E402


def show(inv: Investigation) -> None:
    r = inv.result
    print(f"\n=== {inv.incident_id} · memory {'ON' if inv.memory_enabled else 'OFF'} "
          f"· pattern {inv.detected_pattern or 'unknown'} ===")
    if inv.memory_warning:
        print(f"WARNING: {inv.memory_warning}")
    print(f"Likely cause [{r.confidence}]: {r.likely_cause}")
    if r.no_relevant_history:
        print("NO RELEVANT HISTORY FOUND — generic best-effort answer")
    for line in r.evidence:
        print(f"  evidence: {line}")
    print("Ranked fixes:")
    for i, fix in enumerate(r.ranked_fixes, 1):
        badge = f"history: {fix.past_outcomes}" if fix.based_on_history else "no history"
        print(f"  {i}. {fix.fix} [{badge}]")
        print(f"     {fix.rationale}")
    if inv.memories:
        print("Memories used:")
        for m in inv.memories:
            print(f"  - {m.text[:180]}...")
    print(f"Model: {inv.llm_model} ({inv.llm_latency_ms}ms), backend: {inv.memory_backend}")
    if inv.used_deterministic_fallback:
        print(f"Fallback: {inv.fallback_reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incident_id")
    parser.add_argument("--memory", choices=["on", "off"], default="on")
    parser.add_argument("--compare", action="store_true", help="run memory OFF and ON")
    parser.add_argument("--json", action="store_true", help="print raw Investigation JSON")
    args = parser.parse_args()

    settings = load_settings()
    world = SimulatedWorld(settings.data_dir)
    if world.incident(args.incident_id) is None:
        print(f"Unknown incident. Choose one of: "
              f"{', '.join(i['id'] for i in world.evals()[:5])} ...")
        return 2

    memory, warning = build_memory(settings)
    agent = InvestigationAgent(settings, world=world, memory=memory, memory_warning=warning)
    modes = [False, True] if args.compare else [args.memory == "on"]
    runs = [agent.investigate(args.incident_id, memory_enabled=mode) for mode in modes]

    if args.json:
        payload = [run.model_dump(mode="json") for run in runs]
        print(json.dumps(payload if args.compare else payload[0], indent=2))
    else:
        for run in runs:
            show(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
