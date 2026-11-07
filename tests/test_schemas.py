from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from oncall.schemas import (
    FIX_IDS,
    AgentResult,
    EvalIncident,
    Incident,
    OutcomeRecord,
    RankedFix,
)


def minimal_agent_result(**overrides):
    payload = {
        "likely_cause": "Redis pool exhausted by the deploy",
        "confidence": "medium",
        "evidence": ["ConnectionPoolTimeoutError in checkout-service log"],
        "similar_incidents": [{"id": "INC-014", "why_similar": "same signature"}],
        "ranked_fixes": [{
            "fix": "rollback_deploy_and_increase_pool",
            "rationale": "worked twice on this pattern",
            "based_on_history": True,
            "past_outcomes": "worked 2/2 | failed 0 | partial 0",
        }],
        "no_relevant_history": False,
    }
    payload.update(overrides)
    return payload


def test_agent_result_accepts_controlled_vocabulary():
    result = AgentResult.model_validate(minimal_agent_result())
    assert result.ranked_fixes[0].fix in FIX_IDS


def test_agent_result_rejects_unknown_fix():
    payload = minimal_agent_result()
    payload["ranked_fixes"][0]["fix"] = "give_it_a_kick"
    with pytest.raises(ValidationError):
        AgentResult.model_validate(payload)


def test_agent_result_rejects_duplicate_fixes():
    payload = minimal_agent_result()
    payload["ranked_fixes"].append(dict(payload["ranked_fixes"][0]))
    with pytest.raises(ValidationError):
        AgentResult.model_validate(payload)


def test_agent_result_rejects_bad_confidence_and_id():
    with pytest.raises(ValidationError):
        AgentResult.model_validate(minimal_agent_result(confidence="certain"))
    bad_id = minimal_agent_result(
        similar_incidents=[{"id": "banana", "why_similar": "x"}]
    )
    with pytest.raises(ValidationError):
        AgentResult.model_validate(bad_id)


def test_incident_outcome_vocabulary_enforced():
    with pytest.raises(ValidationError):
        Incident.model_validate({
            "id": "INC-999",
            "family": "memory_leak",
            "service": "search-service",
            "opened_at": "2026-01-01T00:00:00Z",
            "alert": "x",
            "log_file": "",
            "metrics_file": "",
            "fixes_tried": [{"fix": "restart_service", "outcome": "maybe"}],
        })


def test_eval_incident_requires_true_fix_from_vocabulary():
    base = {
        "id": "INC-999",
        "family": "memory_leak",
        "service": "search-service",
        "opened_at": "2026-01-01T00:00:00Z",
        "alert": "x",
        "log_file": "",
        "metrics_file": "",
    }
    assert EvalIncident.model_validate({**base, "true_fix": "rollback_deploy"})
    with pytest.raises(ValidationError):
        EvalIncident.model_validate({**base, "true_fix": "nope"})
    with pytest.raises(ValidationError):
        EvalIncident.model_validate(base)


def test_outcome_record_defaults_to_utc():
    record = OutcomeRecord(incident_id="INC-1", service="auth-service",
                           fix="rollback_deploy", outcome="worked")
    assert record.recorded_at.tzinfo is not None
    assert record.recorded_at.utcoffset().total_seconds() == 0


def test_ranked_fix_requires_known_id():
    with pytest.raises(ValidationError):
        RankedFix(fix="turn_it_off_and_on_again", rationale="x")
