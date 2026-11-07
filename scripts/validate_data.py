"""Validate the simulated world's internal consistency.

A plain script, not a test suite: it reads `data/` and prints one PASS/FAIL line
per invariant, then exits non-zero if anything failed. Run it after regenerating
the world, and before a demo or a benchmark run.

  python scripts/validate_data.py
  python scripts/validate_data.py --quiet     # only failures and the summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oncall.schemas import FIX_IDS, EvalIncident, Incident, Service  # noqa: E402
from oncall.tools import detect_pattern  # noqa: E402

DATA = ROOT / "data"


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Validator:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passes += 1
            print(f"PASS  {name}")
        else:
            self.failures.append(name)
            print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))
        return ok

    def group(self, name: str, problems: list[str], expected: int) -> bool:
        """One PASS/FAIL line for a per-incident invariant across `expected` records."""
        if problems:
            shown = "; ".join(problems[:4]) + ("…" if len(problems) > 4 else "")
            return self.check(name, False, f"{len(problems)}/{expected} bad: {shown}")
        return self.check(name, True, f"{expected}/{expected} records ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print failures and the summary only")
    parser.parse_args(argv)

    services_raw = _load("services.json")["services"]
    services = {s["name"]: Service.model_validate(s) for s in services_raw}
    families = _load("families.json")["families"]
    deploys = {d["id"]: d for d in _load("deploys.json")["deploys"]}
    history = _load("incidents_history.json")["incidents"]
    evals = _load("incidents_eval.json")["incidents"]
    eval_answers = _load("incidents_eval_answers.json")["answers"]
    records = history + evals

    v = Validator()
    print(f"Validating {len(records)} incidents against {len(services)} services, "
          f"{len(deploys)} deploys, {len(families)} families\n")

    # --- shape -------------------------------------------------------------
    v.check("services: 5 records with owner, runtime and SLOs",
            len(services) == 5 and all(
                s.team and s.runtime and s.primary_oncall and s.slo_p95_ms > 0
                and s.slo_error_rate_pct > 0 for s in services.values()))

    v.check("history: 28 resolved incidents covering the 5 seeded families",
            len(history) == 28 and {h["family"] for h in history} >= {
                "redis_pool_exhaustion", "memory_leak", "expired_tls_cert",
                "slow_query_migration", "payments_rate_limit"}
            and {h["family"] for h in history} <= set(families))

    problems = []
    for raw in history:
        try:
            inc = Incident.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{raw.get('id')}: {exc}")
            continue
        if inc.status != "resolved" or not inc.root_cause or not inc.resolved_by:
            problems.append(f"{inc.id}: incomplete postmortem")
        elif sum(1 for f in inc.fixes_tried if f.outcome == "worked") != 1:
            problems.append(f"{inc.id}: not exactly one working fix")
    v.group("history: every record is a complete postmortem with exactly one working fix",
            problems, len(history))

    v.check("eval: public records contain no true_fix and hidden answers match their ids",
            all("true_fix" not in raw for raw in evals)
            and set(eval_answers) == {raw["id"] for raw in evals}
            and all(value in FIX_IDS for value in eval_answers.values()))

    problems = []
    for raw in evals:
        try:
            inc = EvalIncident.model_validate({**raw, "true_fix": eval_answers.get(raw["id"], "")})
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{raw.get('id')}: {exc}")
            continue
        if inc.status != "open":
            problems.append(f"{inc.id}: status is {inc.status}")
        elif inc.root_cause or inc.fixes_tried or inc.resolved_by:
            problems.append(f"{inc.id}: carries a resolution")
    v.group("eval: 20 open incidents with no root cause, fixes tried or resolver recorded",
            problems, len(evals))

    v.check("eval: exactly 2 incidents marked novel",
            sum(1 for i in evals if i.get("novel")) == 2)

    # --- the answer must not be reachable from agent-visible text ----------
    # Ground truth is stored separately from public eval records, and nothing
    # agent-visible (alert, symptoms, log extract) may hint at it by identifier.
    problems = []
    for raw in evals:
        inc = EvalIncident.model_validate({**raw, "true_fix": eval_answers.get(raw["id"], "")})
        log_path = ROOT / inc.log_file
        visible = " ".join([inc.alert, *inc.symptoms,
                            log_path.read_text(encoding="utf-8") if log_path.exists() else ""])
        leaked = [fix for fix in sorted(FIX_IDS) if fix in visible]
        if leaked:
            problems.append(f"{inc.id}: log/alert mentions {leaked}")
    v.group("eval: no fix-vocabulary id appears in the alert, symptoms or log extract",
            problems, len(evals))

    # --- cross-file consistency -------------------------------------------
    ids = [i["id"] for i in records]
    v.check("incident ids are unique across history and eval", len(ids) == len(set(ids)))

    problems = []
    for raw in records:
        inc_id = raw["id"]
        signature = families[raw["family"]]["signature"]
        log_path = ROOT / raw["log_file"]
        if not log_path.exists():
            problems.append(f"{inc_id}: missing log file")
            continue
        if signature not in log_path.read_text(encoding="utf-8"):
            problems.append(f"{inc_id}: signature {signature!r} not in log")
    v.group("every incident log exists and contains its family signature",
            problems, len(records))

    problems = []
    for raw in records:
        inc_id = raw["id"]
        metrics_path = ROOT / raw["metrics_file"]
        if not metrics_path.exists():
            problems.append(f"{inc_id}: missing metrics file")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("incident_id") != inc_id:
            problems.append(f"{inc_id}: metrics name the wrong incident")
            continue
        times = [_dt(point["t"]) for point in metrics.get("series", [])]
        if times != sorted(times):
            problems.append(f"{inc_id}: metric points out of order")
    v.group("every metrics snapshot exists, matches its incident and is time-ordered",
            problems, len(records))

    problems = []
    for raw in records:
        deploy_id = raw.get("related_deploy_id")
        if not deploy_id:
            continue
        deploy = deploys.get(deploy_id)
        if deploy is None:
            problems.append(f"{raw['id']}: deploy {deploy_id} is dangling")
            continue
        if _dt(deploy["deployed_at"]) >= _dt(raw["opened_at"]):
            problems.append(f"{raw['id']}: deploy is not before the alert")
            continue
        dependency_of = raw["service"] in services and (
            deploy["service"] == raw["service"]
            or deploy["service"] in services[raw["service"]].dependencies)
        if not dependency_of:
            problems.append(f"{raw['id']}: deploy belongs to an unrelated service")
    v.group("every referenced deploy exists, predates the alert and is related",
            problems, len([r for r in records if r.get("related_deploy_id")]))

    v.check("deploys: ids unique with version, summary and commit",
            all(d["service"] and d["version"] and d["commit"] for d in deploys.values()))

    # --- the recall gate is reproducible from the data alone ---------------
    problems = []
    for raw in records:
        log_path = ROOT / raw["log_file"]
        text = " ".join([raw["alert"], *raw.get("symptoms", []),
                         log_path.read_text(encoding="utf-8") if log_path.exists() else ""])
        detected, _score = detect_pattern(text, families)
        if detected != raw["family"]:
            problems.append(f"{raw['id']}: detected {detected!r}, stored {raw['family']!r}")
    v.group("pattern detection over alert+symptoms+log matches the stored family for every incident",
            problems, len(records))

    # --- the demo's headline claim ----------------------------------------
    traps = [raw for raw in history
             if any(f["fix"] == "restart_service" and f["outcome"] == "failed" for f in raw["fixes_tried"])
             and any(f["outcome"] == "worked" and f["fix"] != "restart_service"
                     for f in raw["fixes_tried"])]
    v.check(f"history contains restart-is-the-wrong-move traps ({len(traps)} found)", len(traps) >= 3)

    seen: dict[str, set[str]] = {}
    for raw in history:
        for fix in raw["fixes_tried"]:
            seen.setdefault(fix["fix"], set()).add(fix["outcome"])
    conflicting = sorted(f for f, outcomes in seen.items() if {"worked", "failed"} <= outcomes)
    v.check(f"history contains genuinely conflicting outcomes ({', '.join(conflicting) or 'none'})",
            bool(conflicting))

    print()
    if v.failures:
        print(f"FAIL  {len(v.failures)} of {len(v.failures) + v.passes} checks failed:")
        for name in v.failures:
            print(f"  - {name}")
        return 1
    print(f"PASS  all {v.passes} data-consistency checks passed "
          f"({len(records)} incidents, {len(deploys)} deploys, {len(services)} services).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
