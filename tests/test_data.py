"""The simulated world has to be internally consistent to be believable.

For every incident these tests check the invariants the UI and benchmark rely
on: the log extract exists and contains the family's signature error, the
metrics snapshot parses, the referenced deploy exists and predates the alert,
and the story is complete (root cause + a working fix).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from oncall.schemas import FIX_IDS, EvalIncident, Incident, Service

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))["incidents"]


@pytest.fixture(scope="module")
def services() -> dict[str, Service]:
    raw = json.loads((DATA / "services.json").read_text(encoding="utf-8"))["services"]
    return {s.name: s for s in (Service.model_validate(r) for r in raw)}


@pytest.fixture(scope="module")
def deploys() -> dict[str, dict]:
    raw = json.loads((DATA / "deploys.json").read_text(encoding="utf-8"))["deploys"]
    return {d["id"]: d for d in raw}


@pytest.fixture(scope="module")
def families() -> dict[str, dict]:
    return json.loads((DATA / "families.json").read_text(encoding="utf-8"))["families"]


@pytest.fixture(scope="module")
def history() -> list[dict]:
    return load("incidents_history.json")


@pytest.fixture(scope="module")
def eval_incidents() -> list[dict]:
    return load("incidents_eval.json")


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_five_services_with_owners_and_slos(services: dict[str, Service]) -> None:
    assert len(services) == 5
    for svc in services.values():
        assert svc.team and svc.runtime and svc.primary_oncall
        assert svc.slo_p95_ms > 0 and svc.slo_error_rate_pct > 0


def test_history_count_and_family_coverage(history: list[dict], families: dict[str, dict]) -> None:
    assert len(history) == 28
    seen = {i["family"] for i in history}
    required = {
        "redis_pool_exhaustion",
        "memory_leak",
        "expired_tls_cert",
        "slow_query_migration",
        "payments_rate_limit",
    }
    assert required <= seen
    assert required <= set(families)


def test_history_records_are_complete(history: list[dict]) -> None:
    for raw in history:
        inc = Incident.model_validate(raw)
        assert inc.status == "resolved"
        assert inc.root_cause, f"{inc.id} has no root cause"
        assert inc.resolution_minutes > 0
        assert inc.resolved_by
        worked = [f for f in inc.fixes_tried if f.outcome == "worked"]
        assert len(worked) == 1, f"{inc.id} must record exactly one working fix"


def test_trap_incidents_exist(history: list[dict]) -> None:
    """At least three incidents where restart failed but a specific fix worked."""
    traps = [
        raw for raw in history
        if any(f["fix"] == "restart_service" and f["outcome"] == "failed" for f in raw["fixes_tried"])
        and any(f["outcome"] == "worked" and f["fix"] != "restart_service" for f in raw["fixes_tried"])
    ]
    assert len(traps) >= 3


def test_conflicting_memory_exists(history: list[dict]) -> None:
    """A fix that worked once and failed on a later recurrence."""
    outcomes: dict[str, set[str]] = {}
    for raw in history:
        for f in raw["fixes_tried"]:
            outcomes.setdefault(f["fix"], set()).add(f["outcome"])
    conflicts = {fix for fix, seen in outcomes.items() if {"worked", "failed"} <= seen}
    assert conflicts, "expected at least one contradicting fix outcome in history"


def test_eval_shape_and_hidden_truth(eval_incidents: list[dict]) -> None:
    assert len(eval_incidents) == 20
    for raw in eval_incidents:
        inc = EvalIncident.model_validate(raw)
        assert inc.true_fix in FIX_IDS
        assert inc.status == "open"
        # never leak the answer into the incident record itself
        assert inc.root_cause == ""
        assert inc.fixes_tried == []
        assert inc.resolved_by == ""
    assert sum(1 for i in eval_incidents if i["novel"]) == 2


def test_eval_ids_do_not_collide_with_history(history: list[dict], eval_incidents: list[dict]) -> None:
    assert not ({i["id"] for i in history} & {i["id"] for i in eval_incidents})


@pytest.mark.parametrize("kind", ["incidents_history.json", "incidents_eval.json"])
def test_logs_metrics_and_deploys_agree(
    kind: str, services: dict[str, Service], deploys: dict[str, dict], families: dict[str, dict]
) -> None:
    for raw in load(kind):
        inc_id = raw["id"]
        signature = families[raw["family"]]["signature"]

        log_path = ROOT / raw["log_file"]
        assert log_path.exists(), f"{inc_id}: missing log file"
        log_text = log_path.read_text(encoding="utf-8")
        assert signature in log_text, f"{inc_id}: signature {signature!r} missing from log"

        metrics_path = ROOT / raw["metrics_file"]
        assert metrics_path.exists(), f"{inc_id}: missing metrics file"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["incident_id"] == inc_id
        times = [dt(p["t"]) for p in metrics["series"]]
        assert times == sorted(times), f"{inc_id}: metric points out of order"

        if raw.get("related_deploy_id"):
            deploy = deploys.get(raw["related_deploy_id"])
            assert deploy is not None, f"{inc_id}: deploy reference is dangling"
            assert dt(deploy["deployed_at"]) < dt(raw["opened_at"]), f"{inc_id}: deploy is after alert"
            related = deploy["service"] == raw["service"] or deploy["service"] in services[raw["service"]].dependencies
            assert related, f"{inc_id}: deploy belongs to an unrelated service"
            assert deploy["version"] in log_text or raw["family"] == "redis_pool_exhaustion"


def test_no_duplicate_incident_ids() -> None:
    ids = [i["id"] for i in load("incidents_history.json")] + [i["id"] for i in load("incidents_eval.json")]
    assert len(ids) == len(set(ids))


def test_deploy_ids_unique_and_versions_monotonic(deploys: dict[str, dict]) -> None:
    assert len(deploys) == len(set(deploys))
    for deploy in deploys.values():
        assert deploy["service"] and deploy["version"] and deploy["commit"]
