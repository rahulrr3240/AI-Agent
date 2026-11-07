import json
from pathlib import Path
from unittest.mock import patch

import pytest

from oncall.agent import InvestigationAgent, record_outcome
from oncall.config import Settings
from oncall.memory import LocalMemory
from oncall.schemas import AgentResult, Incident, RankedFix
from oncall.tools import SimulatedWorld, detect_pattern

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def world() -> SimulatedWorld:
    return SimulatedWorld(DATA)


@pytest.fixture(scope="module")
def memory(world, tmp_path_factory) -> LocalMemory:
    store = tmp_path_factory.mktemp("memory")
    mem = LocalMemory(store, "agent-test")
    for raw in world.history():
        mem.retain_incident(Incident.model_validate(raw))
    return mem


@pytest.fixture(scope="module")
def agent(world, memory, tmp_path_factory) -> InvestigationAgent:
    settings = Settings(groq_api_key="", hindsight_base_url="",
                        base_dir=tmp_path_factory.mktemp("base"), data_dir=DATA)
    return InvestigationAgent(settings, world=world, memory=memory,
                              memory_warning="local keyword fallback (test)")


def test_pattern_detection_matches_every_seeded_family(world):
    hits = 0
    incidents = world.history() + world.evals()
    for raw in incidents:
        text = (raw["alert"] + " " + " ".join(raw["symptoms"]) + " "
                + world.get_logs(raw["service"], raw["id"])["text"])
        family, score = detect_pattern(text, world.families())
        hits += family == raw["family"] and score >= 2
    assert hits == len(incidents)


def test_memory_on_beats_generic_restart(agent):
    on = agent.investigate("INC-103", memory_enabled=True)
    off = agent.investigate("INC-103", memory_enabled=False)

    assert on.result.ranked_fixes[0].fix == "renew_certificate"
    assert on.result.ranked_fixes[0].based_on_history
    assert on.result.similar_incidents
    assert on.memories and on.detected_pattern == "expired_tls_cert"

    assert off.result.ranked_fixes[0].fix == "restart_service"
    assert off.memories == []
    assert off.result.similar_incidents == []
    assert off.used_deterministic_fallback


def test_novel_incident_reports_no_relevant_history(agent):
    result = agent.investigate("INC-119", memory_enabled=True)
    assert result.result.no_relevant_history is True
    assert result.result.similar_incidents == []
    assert result.result.confidence == "low"
    assert result.memories == []
    assert "not determined" in result.result.likely_cause


def test_conflicting_memory_lowers_confidence(agent):
    result = agent.investigate("INC-106", memory_enabled=True)
    assert "increase_pool_size" in result.conflicts
    assert any("Conflicting memory" in line for line in result.result.evidence)
    assert result.result.confidence == "medium"
    # The current incident has no deploy; the cause must be attributed to the
    # remembered incident rather than asserted about this one.
    assert "Closest recorded incident" in result.result.likely_cause


def test_llm_prompt_never_contains_hidden_ground_truth(agent, world):
    raw = next(i for i in world.evals() if i["id"] == "INC-101")
    inc = Incident.model_validate(raw)
    logs = world.get_logs(inc.service, inc.id)
    deploys = world.get_recent_deploys(inc.service, 6, as_of=inc.opened_at)
    metrics = world.get_metrics(inc.service, inc.id)
    memories = agent.memory.recall_similar(inc.alert, inc.service, inc.symptoms,
                                           world.log_excerpt(inc.id), k=5)
    messages = agent._build_messages(inc, logs, deploys, metrics, memories, True)
    text = json.dumps(messages)
    assert "true_fix" not in text
    assert '"root_cause"' not in text
    assert '"resolved_by"' not in text
    # The hidden answer must not be presented as ground truth. It may appear
    # only inside the fix vocabulary (a list of valid ids) or in recalled
    # history, both of which are legitimate.


def test_guardrail_demotes_failed_fix_and_surfaces_known_good(agent):
    history = {
        "restart_service": {"worked": 0, "failed": 3, "partial": 0,
                            "incidents": {"INC-1"}, "services": {"x"}},
        "renew_certificate": {"worked": 3, "failed": 0, "partial": 0,
                              "incidents": {"INC-2"}, "services": {"x"}},
    }
    ranked = [
        RankedFix(fix="restart_service", rationale="obvious", based_on_history=True),
        RankedFix(fix="scale_up", rationale="maybe", based_on_history=False),
    ]
    ordered = agent._apply_memory_guardrail(ranked, history)
    assert ordered[0].fix == "renew_certificate"
    assert ordered[-1].fix == "restart_service"

    inserted = agent._apply_memory_guardrail(
        [RankedFix(fix="scale_up", rationale="x")], history)
    assert inserted[0].fix == "renew_certificate"


def test_postprocess_filters_hallucinated_incident_ids(agent):
    hits = [type("H", (), {"text": "INC-014 pool", "metadata": {"incident_id": "INC-014"}})()]
    result = AgentResult.model_validate({
        "likely_cause": "x",
        "confidence": "medium",
        "evidence": ["e"],
        "similar_incidents": [
            {"id": "INC-014", "why_similar": "same"},
            {"id": "INC-777", "why_similar": "invented"},
        ],
        "ranked_fixes": [{"fix": "restart_service", "rationale": "x"}],
        "no_relevant_history": False,
    })
    processed = agent._postprocess(result, hits, {}, True, [])
    assert [s.id for s in processed.similar_incidents] == ["INC-014"]


def test_empty_logs_lower_confidence(agent):
    agent_obj = InvestigationAgent(agent.settings, world=agent.world,
                                   memory=agent.memory, memory_warning=agent.memory_warning)
    with patch.object(agent_obj.world, "get_logs", return_value={
        "text": "", "line_count": 0, "error_lines": 0, "missing": False, "path": ""
    }):
        result = agent_obj.investigate("INC-104", memory_enabled=True)
    assert result.result.confidence == "low"
    assert any("Log extract was empty" in line for line in result.result.evidence)


def test_record_outcome_is_deduplicated(agent):
    world = agent.world
    inc = world.incident("INC-101")
    record, doc_id = record_outcome(agent.memory, inc, "rollback_deploy_and_increase_pool",
                                    "worked", "confirmed on call")
    record2, doc_id2 = record_outcome(agent.memory, inc, "rollback_deploy_and_increase_pool",
                                      "worked", "duplicate click")
    assert doc_id == doc_id2
    assert record.family == "redis_pool_exhaustion"
    outcomes = [d for d in agent.memory._docs.values()
                if d["metadata"].get("record_type") == "outcome"]
    assert len([o for o in outcomes if o["id"] == doc_id]) == 1
