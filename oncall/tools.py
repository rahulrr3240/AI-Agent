"""The investigation tools. They read only the simulated files in data/.

No network calls and no real infrastructure: `get_logs`, `get_recent_deploys`
and `get_metrics` are deterministic readers over the generated world.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from oncall.config import BASE_DIR, DATA_DIR


def _parse(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class SimulatedWorld:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir or DATA_DIR)
        self._cache: dict[str, Any] = {}

    # -- loaders -----------------------------------------------------------
    def _json(self, name: str) -> dict:
        if name not in self._cache:
            path = self.data_dir / name
            self._cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[name]

    def services(self) -> dict[str, dict]:
        return {s["name"]: s for s in self._json("services.json")["services"]}

    def families(self) -> dict[str, dict]:
        return self._json("families.json")["families"]

    def history(self) -> list[dict]:
        return self._json("incidents_history.json")["incidents"]

    def evals(self, *, include_truth: bool = False) -> list[dict]:
        incidents = self._json("incidents_eval.json")["incidents"]
        if not include_truth:
            return incidents
        answers = self._json("incidents_eval_answers.json")["answers"]
        return [{**incident, "true_fix": answers[incident["id"]]} for incident in incidents]

    def deploys(self) -> list[dict]:
        return self._json("deploys.json")["deploys"]

    def incident(self, incident_id: str) -> dict | None:
        for item in self.history() + self.evals():
            if item["id"] == incident_id:
                return item
        return None

    def eval_incident(self, incident_id: str) -> dict | None:
        for item in self.evals():
            if item["id"] == incident_id:
                return item
        return None

    # -- tools -------------------------------------------------------------
    def get_logs(self, service: str, incident_id: str, last_n_lines: int = 45) -> dict[str, Any]:
        path = self.data_dir / "logs" / f"{incident_id}.log"
        if not path.exists():
            return {"text": "", "line_count": 0, "error_lines": 0, "missing": True,
                    "path": str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path)}
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not raw:
            return {"text": "", "line_count": 0, "error_lines": 0, "missing": False,
                    "path": f"data/logs/{incident_id}.log"}
        lines = raw[-last_n_lines:]
        error_lines = sum(1 for line in lines if "ERROR" in line or "level=error" in line)
        return {
            "text": "\n".join(lines),
            "line_count": len(lines),
            "error_lines": error_lines,
            "missing": False,
            "path": f"data/logs/{incident_id}.log",
        }

    def get_recent_deploys(self, service: str, hours: float = 6,
                           as_of: datetime | str | None = None) -> list[dict[str, Any]]:
        reference = _parse(as_of) if as_of else datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        window_start = reference - timedelta(hours=hours)
        deps = set(self.services().get(service, {}).get("dependencies", []))
        found = []
        for deploy in self.deploys():
            deployed = _parse(deploy["deployed_at"])
            if not (window_start <= deployed <= reference):
                continue
            if deploy["service"] == service:
                relation = "own"
            elif deploy["service"] in deps:
                relation = "dependency"
            else:
                continue
            found.append({
                "id": deploy["id"],
                "service": deploy["service"],
                "version": deploy["version"],
                "deployed_at": deploy["deployed_at"],
                "deployed_by": deploy["deployed_by"],
                "summary": deploy["summary"],
                "previous_version": deploy.get("previous_version"),
                "relation": relation,
                "minutes_before_alert": int((reference - deployed).total_seconds() // 60),
            })
        found.sort(key=lambda d: d["deployed_at"], reverse=True)
        return found

    def get_metrics(self, service: str, incident_id: str) -> dict[str, Any]:
        path = self.data_dir / "metrics" / f"{incident_id}.json"
        if not path.exists():
            return {"available": False, "reason": f"no metrics snapshot for {incident_id}"}
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["available"] = True
        return payload

    def log_excerpt(self, incident_id: str, chars: int = 900) -> str:
        logs = self.get_logs("", incident_id, last_n_lines=30)
        text = " ".join(logs["text"].split())
        return text[-chars:]


def detect_pattern(text: str, families: dict[str, dict]) -> tuple[str | None, int]:
    """Keyword-classify the failure signature in alert + symptoms + logs.

    This is deliberately transparent runbook logic (substring matches from
    data/families.json), not a peek at the incident's stored family label. It
    is used to keep recall precise — a novel alert must be able to come back
    with no relevant history instead of a superficially similar old incident.
    """
    haystack = " " + " ".join(text.lower().split()) + " "
    best_family: str | None = None
    best_score = 0
    for family, meta in families.items():
        score = sum(haystack.count(keyword.lower()) for keyword in meta.get("keywords", []))
        if score > best_score:
            best_family, best_score = family, score
    return (best_family, best_score) if best_score >= 2 else (None, best_score)


def tool_descriptions() -> list[dict[str, Any]]:
    """JSON schemas used by agentic (function-calling) mode."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_logs",
                "description": "Read the most recent log lines for this incident.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "last_n_lines": {"type": "integer", "description": "How many trailing lines (10-80)."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recent_deploys",
                "description": "List deploys of this service (and its dependencies) in the hours before the alert.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hours": {"type": "integer", "description": "Lookback window in hours (1-24)."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_metrics",
                "description": "Read the metric snapshot (latency, error rate, resource pressure) for this incident.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memory",
                "description": "Search incident memory for similar past incidents and fix outcomes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {"type": "string", "description": "What to look for, e.g. the failure signature."},
                    },
                    "required": ["focus"],
                },
            },
        },
    ]
