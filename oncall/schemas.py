"""Pydantic schemas shared by data, agent, memory and evaluation.

The fix vocabulary is a controlled list on purpose: the same identifier is used
in simulated ground truth, in memories written to Hindsight, and in the agent's
output, which makes evaluation deterministic and unambiguous.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --- controlled vocabulary -------------------------------------------------
FIX_LABELS: dict[str, str] = {
    "restart_service": "Restart the service",
    "rollback_deploy": "Roll back the last deploy",
    "rollback_deploy_and_increase_pool": "Roll back deploy + raise Redis pool size",
    "increase_pool_size": "Raise Redis connection pool size",
    "renew_certificate": "Renew and reload the TLS certificate",
    "add_db_index": "Add the missing database index",
    "enable_backoff_circuit_breaker": "Enable exponential backoff + circuit breaker",
    "scale_up": "Scale up replicas / consumers",
    "clear_cache": "Flush the cache",
    "rollback_secret_rotation": "Roll back the signing-key rotation",
}
FIX_IDS = frozenset(FIX_LABELS)

FIX_OUTCOMES = ("worked", "failed", "partial")
CONFIDENCE_LEVELS = ("low", "medium", "high")


def is_valid_fix(fix_id: str) -> bool:
    return fix_id in FIX_IDS


# --- simulated world -------------------------------------------------------
class Service(BaseModel):
    name: str
    team: str
    runtime: str
    dependencies: list[str] = Field(default_factory=list)
    slo_p95_ms: int
    slo_error_rate_pct: float
    description: str
    primary_oncall: str


class Deploy(BaseModel):
    id: str
    service: str
    version: str
    deployed_at: datetime
    deployed_by: str
    summary: str
    commit: str
    previous_version: str | None = None


class FixTried(BaseModel):
    fix: str
    outcome: Literal["worked", "failed", "partial"]
    note: str = ""

    @field_validator("fix")
    @classmethod
    def _known_fix(cls, v: str) -> str:
        if v not in FIX_IDS:
            raise ValueError(f"unknown fix id: {v!r}")
        return v


class Incident(BaseModel):
    id: str
    family: str
    service: str
    opened_at: datetime
    alert: str
    symptoms: list[str] = Field(default_factory=list)
    severity: Literal["P1", "P2", "P3"] = "P2"
    status: Literal["open", "resolved"] = "resolved"
    root_cause: str = ""
    fixes_tried: list[FixTried] = Field(default_factory=list)
    resolution_minutes: int = 0
    resolved_by: str = ""
    related_deploy_id: str | None = None
    log_file: str
    metrics_file: str
    tags: list[str] = Field(default_factory=list)


class EvalIncident(Incident):
    """An unresolved incident with a hidden ground-truth fix.

    `true_fix` is used only by the simulator and the benchmark. It is never
    included in agent context nor retained before an outcome is recorded.
    """

    true_fix: str
    novel: bool = False

    @field_validator("true_fix")
    @classmethod
    def _known_true_fix(cls, v: str) -> str:
        if v not in FIX_IDS:
            raise ValueError(f"unknown true_fix id: {v!r}")
        return v


# --- agent I/O -------------------------------------------------------------
class SimilarIncident(BaseModel):
    id: str
    why_similar: str

    @field_validator("id")
    @classmethod
    def _looks_like_id(cls, v: str) -> str:
        """Shape check only. The real gate is `_postprocess`, which drops any cited
        id that is not in the recalled memories — so a live incident resolved last
        week (LIVE-…) can be cited, and an invented one still cannot."""
        if not v.strip().upper().startswith(("INC-", "LIVE-")):
            raise ValueError(f"similar incident id must look like INC-### or LIVE-…: {v!r}")
        return v


class RankedFix(BaseModel):
    fix: str
    rationale: str
    based_on_history: bool = False
    past_outcomes: str = "none"

    @field_validator("fix")
    @classmethod
    def _known_fix(cls, v: str) -> str:
        if v not in FIX_IDS:
            raise ValueError(f"unknown fix id: {v!r}")
        return v


class AgentResult(BaseModel):
    likely_cause: str
    confidence: Literal["low", "medium", "high"]
    evidence: list[str] = Field(default_factory=list)
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    ranked_fixes: list[RankedFix] = Field(min_length=1, max_length=5)
    no_relevant_history: bool = False

    @field_validator("ranked_fixes")
    @classmethod
    def _unique_fixes(cls, v: list[RankedFix]) -> list[RankedFix]:
        seen: set[str] = set()
        for item in v:
            if item.fix in seen:
                raise ValueError(f"duplicate ranked fix: {item.fix}")
            seen.add(item.fix)
        return v


class StepEvent(BaseModel):
    name: str
    label: str
    status: Literal["running", "done", "error", "skipped"] = "done"
    detail: str = ""
    duration_ms: int = 0


class MemoryHit(BaseModel):
    id: str = ""
    text: str
    score: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    occurred: str | None = None


class Investigation(BaseModel):
    incident_id: str
    memory_enabled: bool
    detected_pattern: str | None = None
    result: AgentResult
    steps: list[StepEvent] = Field(default_factory=list)
    memories: list[MemoryHit] = Field(default_factory=list)
    memory_backend: str = "hindsight"
    memory_warning: str | None = None
    llm_model: str = "none"
    llm_latency_ms: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    used_deterministic_fallback: bool = False
    fallback_reason: str | None = None
    conflicts: list[str] = Field(default_factory=list)


class OutcomeRecord(BaseModel):
    incident_id: str
    service: str
    family: str = ""
    alert: str = ""
    fix: str
    outcome: Literal["worked", "failed", "partial"]
    note: str = ""
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
