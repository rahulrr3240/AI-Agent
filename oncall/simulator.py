"""A simulated on-call engineer.

Given the agent's ranked fixes and the incident's hidden `true_fix`, the
engineer tries fixes in order until the real one is reached. Wrong attempts
cost time and are recorded as failures, which is what condition C writes back
into memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from oncall.schemas import EvalIncident, RankedFix

BASE_MINUTES = 6
MINUTES_PER_FAILED_ATTEMPT = 8
MAX_ATTEMPTS = 5


@dataclass
class Attempt:
    fix: str
    outcome: str  # worked | failed
    note: str


@dataclass
class Resolution:
    incident_id: str
    true_fix: str
    attempts: list[Attempt] = field(default_factory=list)
    resolved: bool = False
    top1_correct: bool = False
    attempts_to_resolve: int = 0
    simulated_minutes: int = 0

    @property
    def top_fix(self) -> str:
        return self.attempts[0].fix if self.attempts else ""

    @property
    def tried_fixes(self) -> list[str]:
        return [attempt.fix for attempt in self.attempts]

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "true_fix": self.true_fix,
            "top_fix": self.top_fix,
            "attempts": [{"fix": a.fix, "outcome": a.outcome, "note": a.note}
                         for a in self.attempts],
            "resolved": self.resolved,
            "top1_correct": self.top1_correct,
            "attempts_to_resolve": self.attempts_to_resolve,
            "simulated_minutes": self.simulated_minutes,
        }


class SimulatedEngineer:
    def resolve(self, incident: EvalIncident | dict,
                ranked_fixes: Iterable[RankedFix | str],
                base_minutes: int = BASE_MINUTES,
                minutes_per_failure: int = MINUTES_PER_FAILED_ATTEMPT) -> Resolution:
        inc = incident if isinstance(incident, EvalIncident) else EvalIncident.model_validate(incident)
        ordered: list[str] = []
        for item in ranked_fixes:
            fix = item.fix if isinstance(item, RankedFix) else str(item)
            if fix and fix not in ordered:
                ordered.append(fix)
        ordered = ordered[:MAX_ATTEMPTS]

        result = Resolution(incident_id=inc.id, true_fix=inc.true_fix)
        failures = 0
        for index, fix in enumerate(ordered):
            if fix == inc.true_fix:
                result.attempts.append(Attempt(
                    fix=fix, outcome="worked",
                    note=f"simulated engineer confirmed {fix} resolved {inc.id}",
                ))
                result.resolved = True
                result.attempts_to_resolve = index + 1
                result.simulated_minutes = base_minutes + failures * minutes_per_failure
                break
            failures += 1
            result.attempts.append(Attempt(
                fix=fix, outcome="failed",
                note=f"simulated engineer tried {fix}; the alert persisted",
            ))

        if not result.resolved:
            result.attempts_to_resolve = len(result.attempts)
            result.simulated_minutes = base_minutes + len(result.attempts) * minutes_per_failure

        result.top1_correct = bool(result.attempts) and result.attempts[0].fix == inc.true_fix
        return result


def resolve_with_agent(agent, memory, incident: dict, memory_enabled: bool,
                       engineer: SimulatedEngineer | None = None) -> tuple[Any, Resolution]:
    """Run the agent, then the engineer. Returns (Investigation, Resolution)."""
    investigation = agent.investigate(incident["id"], memory_enabled=memory_enabled)
    resolution = (engineer or SimulatedEngineer()).resolve(
        incident, investigation.result.ranked_fixes)
    return investigation, resolution
