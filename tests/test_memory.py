import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from oncall.config import Settings
from oncall.memory import (
    HindsightMemory,
    LocalMemory,
    MemoryUnavailable,
    build_memory,
    conflicting_fixes,
    fix_history_from_memories,
    past_outcomes_text,
)
from oncall.schemas import Incident, MemoryHit, OutcomeRecord

DATA = Path(__file__).resolve().parent.parent / "data"


def incident(incident_id="INC-014") -> Incident:
    raw = json.loads((DATA / "incidents_history.json").read_text())["incidents"]
    return Incident.model_validate(next(i for i in raw if i["id"] == incident_id))


class NotFound(Exception):
    status = 404


class FakeHindsight:
    def __init__(self, recall_results=None, memories=None):
        self.retains = []
        self.reflects = []
        self.deleted = []
        self.bank_reads = 0
        self.recall_results = recall_results or []
        self.memories = memories or []
        self.created = None

    def get_bank_config(self, bank_id):
        self.bank_reads += 1
        raise NotFound("bank does not exist")

    def create_bank(self, **kwargs):
        self.created = kwargs

    def retain(self, **kwargs):
        self.retains.append(kwargs)
        return SimpleNamespace()

    def recall(self, **kwargs):
        self.last_recall = kwargs
        return SimpleNamespace(results=self.recall_results)

    def reflect(self, **kwargs):
        self.reflects.append(kwargs)
        return SimpleNamespace(text="Recurring redis pool failures on checkout-service (INC-014).")

    def list_memories(self, **kwargs):
        return SimpleNamespace(items=self.memories, total=len(self.memories))

    def delete_bank(self, bank_id):
        self.deleted.append(bank_id)

    def close(self):
        pass


def settings(tmp_path) -> Settings:
    return Settings(hindsight_base_url="http://fake", hindsight_api_key="k",
                    hindsight_bank_id="unit-bank", base_dir=tmp_path,
                    data_dir=DATA, hindsight_timeout=5)


def test_retain_incident_is_self_contained_and_idempotent(tmp_path):
    fake = FakeHindsight()
    memory = HindsightMemory(settings(tmp_path), client=fake)
    document_id = memory.retain_incident(incident())

    assert document_id == "incident-INC-014"
    call = fake.retains[0]
    assert call["document_id"] == "incident-INC-014"
    assert call["context"] == "incident postmortem"
    assert call["timestamp"].year == 2026
    assert "INC-014" in call["content"]
    assert "ConnectionPoolTimeoutError" in call["content"]
    assert "rollback_deploy_and_increase_pool" in call["content"]
    assert call["metadata"]["record_type"] == "incident"
    assert call["metadata"]["outcomes"] == (
        "restart_service:failed;rollback_deploy_and_increase_pool:worked")
    assert fake.created["mission"]


def test_retain_outcome_uses_stable_document_id(tmp_path):
    fake = FakeHindsight()
    memory = HindsightMemory(settings(tmp_path), client=fake)
    record = OutcomeRecord(incident_id="INC-014", service="checkout-service",
                           family="redis_pool_exhaustion", fix="rollback_deploy",
                           outcome="worked", note="confirmed by priya")
    first = memory.retain_outcome(record)
    second = memory.retain_outcome(record)
    assert first == second == "outcome-INC-014-rollback_deploy"
    assert fake.retains[1]["metadata"]["outcome"] == "worked"
    assert "WORKED" in fake.retains[1]["content"]


def test_recall_groups_facts_into_incident_memories(tmp_path):
    def fact(text, incident_id, record_type="incident", **meta):
        metadata = {"incident_id": incident_id, "record_type": record_type,
                    "service": "checkout-service", "family": "redis_pool_exhaustion"}
        metadata.update(meta)
        return SimpleNamespace(
            id=text[:6], text=text, metadata=metadata, scores={"rerank": 0.91},
            document_id=f"incident-{incident_id}", occurred_start="2026-06-02T03:37:00Z",
            mentioned_at="2026-06-02T04:02:00Z",
        )

    fake = FakeHindsight(recall_results=[
        fact("INC-014: checkout-service p95 hit 4.1s.", "INC-014"),
        fact("INC-014: Redis pool exhausted after deploy v2.31.0.", "INC-014"),
        fact("INC-002: restart_service FAILED, latency returned.", "INC-002"),
        fact("INC-101 fix rollback_deploy_and_increase_pool WORKED.", "INC-101",
             record_type="outcome", fix="rollback_deploy_and_increase_pool", outcome="worked"),
    ])
    memory = HindsightMemory(settings(tmp_path), client=fake)
    hits = memory.recall_similar("p95 latency", "checkout-service",
                                 ["ConnectionPoolTimeoutError"], "pool timeout", k=5)

    assert len(hits) == 3
    first = next(h for h in hits if "INC-014" in h.text)
    assert "redis" in first.text.lower()
    assert first.metadata["incident_id"] == "INC-014"
    assert first.score == pytest.approx(0.91)
    assert fake.last_recall["query_timestamp"] is None
    assert "ConnectionPoolTimeoutError" in fake.last_recall["query"]


def test_recall_empty_is_a_normal_answer(tmp_path):
    memory = HindsightMemory(settings(tmp_path), client=FakeHindsight())
    assert memory.recall_similar("kafka lag", "search-service", ["lag"], "consumer lag") == []


def test_stats_and_delete(tmp_path):
    items = [
        SimpleNamespace(metadata={"record_type": "incident", "incident_id": "INC-001"}),
        SimpleNamespace(metadata={"record_type": "incident", "incident_id": "INC-001"}),
        SimpleNamespace(metadata={"record_type": "outcome", "incident_id": "INC-001",
                                  "service": "checkout-service", "fix": "restart_service",
                                  "outcome": "worked"}),
    ]
    fake = FakeHindsight(memories=items)
    memory = HindsightMemory(settings(tmp_path), client=fake)
    stats = memory.stats()
    assert stats["incidents"] == 1
    assert stats["outcomes"] == 1
    assert stats["memories"] == 3
    memory.delete_bank()
    assert fake.deleted == ["unit-bank"]


def test_hindsight_error_is_wrapped():
    class Broken(FakeHindsight):
        def recall(self, **kwargs):
            raise TimeoutError("took too long")

    memory = HindsightMemory(settings(Path(".")), client=Broken())
    with pytest.raises(MemoryUnavailable) as excinfo:
        memory.recall_similar("x", "auth-service", [], "x")
    assert "recall failed" in str(excinfo.value)


def test_factory_falls_back_to_local_without_url(tmp_path):
    memory, warning = build_memory(Settings(base_dir=tmp_path, data_dir=DATA))
    assert memory.name == "local-keyword"
    assert warning and "fallback" in warning


def test_local_memory_matching_and_novelty(tmp_path):
    memory = LocalMemory(tmp_path / "store", "test-bank")
    raw = json.loads((DATA / "incidents_history.json").read_text())["incidents"]
    for item in raw:
        memory.retain_incident(Incident.model_validate(item))

    inc = incident("INC-014")
    hits = memory.recall_similar(inc.alert, inc.service, inc.symptoms,
                                 "ConnectionPoolTimeoutError timeout waiting for connection from pool", k=3)
    assert hits
    assert "INC-014" in [hit.metadata["incident_id"] for hit in hits]
    assert all(hit.metadata["family"] == "redis_pool_exhaustion" for hit in hits)

    # Novelty filtering lives in the agent (it classifies the failure pattern
    # first); the backend itself only ranks. See test_agent for the gate.
    same_service = memory.recall_similar(
        "search-service consumer lag > 100k messages for 10m",
        "search-service",
        ["consumer group lag 184320", "index freshness SLO breached"],
        "consumer group search-indexer lag=184320 partitions=12 rebalance",
        k=3,
    )
    assert all(hit.metadata["service"] == "search-service" for hit in same_service)


def test_local_memory_outcome_upsert_is_idempotent(tmp_path):
    memory = LocalMemory(tmp_path / "store", "test-bank")
    record = OutcomeRecord(incident_id="INC-014", service="checkout-service",
                           family="redis_pool_exhaustion", fix="restart_service",
                           outcome="worked")
    memory.retain_outcome(record)
    memory.retain_outcome(record)
    stats = memory.stats()
    assert stats["outcomes"] == 1


def test_fix_history_and_conflicts():
    hits = [
        MemoryHit(text="a", metadata={"record_type": "incident", "incident_id": "INC-1",
                                      "service": "auth-service",
                                      "outcomes": "increase_pool_size:worked;restart_service:failed"}),
        MemoryHit(text="b", metadata={"record_type": "incident", "incident_id": "INC-2",
                                      "service": "auth-service",
                                      "outcomes": "increase_pool_size:failed"}),
    ]
    history = fix_history_from_memories(hits)
    assert history["increase_pool_size"]["worked"] == 1
    assert history["increase_pool_size"]["failed"] == 1
    assert conflicting_fixes(history) == ["increase_pool_size"]
    assert past_outcomes_text(history["increase_pool_size"]).startswith("worked 1/2")
    assert past_outcomes_text(None) == "none"
