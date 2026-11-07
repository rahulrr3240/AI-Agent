"""Real input path: turn an alert somebody actually received into an Incident.

Everything else in this repository reads the simulated world in `data/`. This
module is the opposite: it takes text, a pasted Alertmanager/Grafana webhook
body and log lines supplied by the engineer, and produces the same `Incident`
object the normal pipeline already consumes, plus a `LiveEvidence` bundle the
agent reads instead of the fixture files.

Nothing here invents evidence. If logs, deploys or metrics were not supplied,
the bundle says so and the agent states the gap in its own evidence list.

The queue lives in `.live_alerts/queue.jsonl` (one JSON object per line, newest
appended last) so the webhook process and Streamlit can be separate programs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oncall.config import BASE_DIR
from oncall.schemas import Incident
from oncall.tools import detect_pattern

QUEUE_PATH = BASE_DIR / ".live_alerts" / "queue.jsonl"
MAX_LOG_CHARS = 40_000
SEVERITY_MAP = {"critical": "P1", "fatal": "P1", "p1": "P1", "high": "P1",
                "warning": "P2", "warn": "P2", "p2": "P2", "medium": "P2",
                "info": "P3", "informational": "P3", "p3": "P3", "none": "P3"}


def new_incident_id(when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return f"LIVE-{when:%Y%m%d-%H%M%S-%f}"


# --------------------------------------------------------------------------
# Evidence the engineer supplied
# --------------------------------------------------------------------------
@dataclass
class LiveEvidence:
    """Logs / deploys / metrics for a live alert — only what was actually given."""

    logs_text: str = ""
    deploys: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] | None = None
    source: str = "manual"          # manual | alertmanager | grafana | file
    origin: str = ""                # where it came from, for the UI

    @property
    def has_logs(self) -> bool:
        return bool(self.logs_text.strip())

    def log_dict(self) -> dict[str, Any]:
        lines = [line for line in self.logs_text.splitlines() if line.strip()]
        errors = sum(1 for line in lines
                     if "ERROR" in line or "error" in line.lower() or "level=error" in line)
        return {
            "text": self.logs_text,
            "line_count": len(lines),
            "error_lines": errors,
            "missing": not self.has_logs,
            "path": "log paste supplied by the engineer",
        }

    def metrics_dict(self) -> dict[str, Any]:
        if not self.metrics:
            return {"available": False, "reason": "no metrics supplied with this live alert"}
        payload = dict(self.metrics)
        payload.setdefault("summary", {})
        payload.setdefault("series", [])
        payload["available"] = True
        return payload

    def gaps(self) -> list[str]:
        """Plain statements about what this alert did *not* come with."""
        missing = []
        if not self.has_logs:
            missing.append("No log extract was supplied with this live alert, so the answer "
                           "rests on the alert text and recalled history only.")
        if not self.deploys:
            missing.append("No deploy history was supplied for this live alert; deploy-related "
                           "causes could not be checked.")
        if not self.metrics:
            missing.append("No metrics were supplied for this live alert; severity and blast "
                           "radius are unverified.")
        return missing


# --------------------------------------------------------------------------
# Webhook payload normalization
# --------------------------------------------------------------------------
def _service_from_labels(labels: dict[str, Any], fallback: str = "") -> str:
    for key in ("service", "service_name", "application", "app", "job", "stack",
                "namespace", "team"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _severity_from(value: Any) -> str:
    if not isinstance(value, str):
        return "P2"
    return SEVERITY_MAP.get(value.strip().lower(), "P2")


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d*", r"\1", text)  # ns -> us precision
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _symptoms_from_lines(text: str, limit: int = 3) -> list[str]:
    out = []
    for line in text.splitlines():
        if "ERROR" in line or "level=error" in line or "fatal" in line.lower():
            cleaned = " ".join(line.split())[:200]
            if cleaned and cleaned not in out:
                out.append(cleaned)
        if len(out) == limit:
            break
    return out


def parse_webhook(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Alertmanager and Grafana webhook bodies -> a list of normalized alert dicts.

    Alertmanager (and Grafana's Alertmanager-compatible formatter) send
    `{"alerts": [{"labels": …, "annotations": …, "startsAt": …}]}`.
    Grafana's legacy formatter sends `{"title", "state", "message", "evalMatches"}`.
    Anything unrecognizable still becomes one alert carrying the raw text, so a
    custom formatter does not silently drop an incident.
    """
    if not isinstance(payload, dict):
        return []

    deploys = _deploys_from_payload(payload)
    alerts = payload.get("alerts")
    normalized: list[dict[str, Any]] = []

    if isinstance(alerts, list) and alerts:
        for item in alerts:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels") or {}
            annotations = item.get("annotations") or {}
            alertname = str(labels.get("alertname") or labels.get("rule") or "alert")
            summary = str(annotations.get("summary") or "").strip()
            description = str(annotations.get("description") or annotations.get("message")
                              or "").strip()
            text = f"{alertname}: {summary}" if summary else alertname
            if description:
                text = f"{text}. {description}"
            normalized.append({
                "service": _service_from_labels(labels),
                "alert": " ".join(text.split())[:600],
                "severity": _severity_from(labels.get("severity") or item.get("severity")),
                "opened_at": _timestamp(item.get("startsAt")),
                "labels": {str(k): str(v) for k, v in labels.items()},
                "source": "alertmanager",
            })
    elif any(key in payload for key in ("title", "message", "evalMatches", "state")):
        matches = payload.get("evalMatches") or []
        detail = str(payload.get("message") or "")
        if isinstance(matches, list) and matches:
            rendered = "; ".join(
                f"{m.get('metric', 'metric')}={m.get('value', '?')}"
                for m in matches if isinstance(m, dict))
            detail = f"{detail} {rendered}".strip()
        tags = payload.get("tags") or {}
        title = " ".join(str(payload.get("title") or "alert").split())
        normalized.append({
            "service": _service_from_labels(tags if isinstance(tags, dict) else {}),
            "alert": f"{title}. {detail}".strip()[:600],
            "severity": _severity_from(payload.get("severity")),
            "opened_at": _timestamp(payload.get("startedAt") or payload.get("timestamp")),
            "labels": {"state": str(payload.get("state", ""))},
            "metrics": ({"summary": {"supplied_eval_matches": matches}}
                        if isinstance(matches, list) and matches else None),
            "source": "grafana",
        })
    else:
        text = json.dumps(payload)[:600]
        normalized.append({"service": _service_from_labels(payload.get("labels", {})
                                                          if isinstance(payload.get("labels"), dict)
                                                          else {}),
                           "alert": f"Unrecognized alert payload: {text}",
                           "severity": _severity_from(payload.get("severity")),
                           "opened_at": None, "labels": {}, "source": "unknown"})

    for record in normalized:
        record["deploys"] = deploys
        log_text = payload.get("logs") or payload.get("log") or ""
        if isinstance(log_text, str):
            record["logs_text"] = log_text[:MAX_LOG_CHARS]
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            record["metrics"] = metrics
    return normalized


def _deploys_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Take deploy records only if the sender actually attached them."""
    raw = payload.get("deploys") or payload.get("recent_deploys")
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("service") or not item.get("version"):
            continue
        out.append({
            "id": str(item.get("id") or f"deploy-{item['service']}-{item['version']}"),
            "service": str(item["service"]),
            "version": str(item["version"]),
            "deployed_at": str(item.get("deployed_at") or item.get("time") or ""),
            "deployed_by": str(item.get("deployed_by") or ""),
            "summary": str(item.get("summary") or ""),
            "previous_version": item.get("previous_version"),
            "relation": "own",
        })
    return out[:10]


# --------------------------------------------------------------------------
# Building an Incident from real input
# --------------------------------------------------------------------------
def build_incident(service: str, alert_text: str, evidence: LiveEvidence,
                   families: dict[str, dict], *, incident_id: str | None = None,
                   opened_at: datetime | None = None, severity: str = "P2",
                   symptoms: list[str] | None = None) -> Incident:
    """A live alert, shaped exactly like a fixture incident for the pipeline.

    There is no ground truth here: `Incident` has no `true_fix`, and the family
    comes from the same transparent keyword gate the simulated incidents use —
    `unclassified` when nothing matches.
    """
    when = opened_at or datetime.now(timezone.utc)
    detected, _score = detect_pattern(f"{alert_text} {evidence.logs_text}", families)
    return Incident(
        id=incident_id or new_incident_id(when),
        family=detected or "unclassified",
        service=service,
        opened_at=when,
        alert=alert_text,
        symptoms=symptoms if symptoms is not None else _symptoms_from_lines(evidence.logs_text),
        severity=severity,  # type: ignore[arg-type]
        status="open",
        log_file="",
        metrics_file="",
        tags=["live", evidence.source],
    )


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------
def _record_from_alert(alert: dict[str, Any], evidence: LiveEvidence,
                       families: dict[str, dict]) -> dict[str, Any]:
    payload_time = alert.get("opened_at")
    incident = build_incident(
        alert.get("service") or "unknown-service", alert.get("alert") or "", evidence, families,
        opened_at=payload_time if isinstance(payload_time, datetime) else None,
        severity=alert.get("severity") or "P2")
    record = {
        "id": incident.id,
        "service": incident.service,
        "alert": incident.alert,
        "severity": incident.severity,
        "family": incident.family,
        "symptoms": incident.symptoms,
        "opened_at": incident.opened_at.isoformat(),
        "source": evidence.source,
        "origin": evidence.origin,
        "logs_text": evidence.logs_text,
        "deploys": evidence.deploys,
        "metrics": evidence.metrics,
        "status": "queued",
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    if alert.get("labels"):
        record["labels"] = alert["labels"]
    return record


def enqueue(alert: dict[str, Any], families: dict[str, dict]) -> dict[str, Any]:
    """Append one normalized alert to the queue file and return its record."""
    evidence = LiveEvidence(
        logs_text=alert.get("logs_text", "") if isinstance(alert.get("logs_text"), str) else "",
        deploys=alert.get("deploys") or [],
        metrics=alert.get("metrics") if isinstance(alert.get("metrics"), dict) else None,
        source=alert.get("source", "unknown"),
        origin=alert.get("origin", ""),
    )
    record = _record_from_alert(alert, evidence, families)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def enqueue_manual(service: str, alert_text: str, logs_text: str, families: dict[str, dict],
                   *, source: str = "manual", origin: str = "", severity: str = "P2",
                   webhook: dict[str, Any] | None = None) -> dict[str, Any]:
    """Queue an alert typed into the UI (optionally after merging a webhook body)."""
    alert = {"service": service, "alert": alert_text, "severity": severity,
             "source": source, "deploys": [], "logs_text": logs_text[:MAX_LOG_CHARS]}
    if webhook:
        parsed = parse_webhook(webhook)
        if parsed:
            first = parsed[0]
            alert["service"] = service or first.get("service") or ""
            alert["alert"] = " ".join(p for p in (alert_text, first.get("alert"),
                                                  first.get("detail", "")) if p).strip()
            alert["severity"] = first.get("severity") or severity
            alert["deploys"] = first.get("deploys") or []
            alert["metrics"] = first.get("metrics")
            alert["logs_text"] = (alert["logs_text"] + "\n" + first.get("logs_text", "")).strip()
            alert["source"] = first.get("source", source)
            alert["labels"] = first.get("labels", {})
            alert["opened_at"] = first.get("opened_at")
    return enqueue(alert, families)


def load_queue(limit: int | None = None) -> list[dict[str, Any]]:
    """Queued live alerts, newest first."""
    if not QUEUE_PATH.exists():
        return []
    records = []
    for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("id"):
            records.append(record)
    records.sort(key=lambda r: r.get("received_at", ""), reverse=True)
    return records[:limit] if limit else records


def get_record(incident_id: str) -> dict[str, Any] | None:
    return next((r for r in load_queue() if r["id"] == incident_id), None)


def incident_from_record(record: dict[str, Any]) -> Incident:
    """Rebuild the live Incident from a queue record (stable across reruns)."""
    opened_at = _timestamp(record.get("opened_at")) or datetime.now(timezone.utc)
    return Incident(
        id=record["id"],
        family=record.get("family") or "unclassified",
        service=record.get("service") or "unknown-service",
        opened_at=opened_at,
        alert=record.get("alert") or "",
        symptoms=record.get("symptoms") or [],
        severity=record.get("severity") or "P2",  # type: ignore[arg-type]
        status=record.get("status") if record.get("status") in ("open", "resolved") else "open",
        log_file="",
        metrics_file="",
        tags=["live", record.get("source", "manual")],
    )


def evidence_from_record(record: dict[str, Any]) -> LiveEvidence:
    return LiveEvidence(
        logs_text=record.get("logs_text") or "",
        deploys=record.get("deploys") or [],
        metrics=record.get("metrics") if isinstance(record.get("metrics"), dict) else None,
        source=record.get("source") or "manual",
        origin=record.get("origin") or "",
    )


def mark_investigated(incident_id: str) -> None:
    record = get_record(incident_id)
    if record and record.get("status") != "investigated":
        record["status"] = "investigated"
        _replace_record(record)


def _replace_record(updated: dict[str, Any]) -> None:
    records = load_queue()
    records = [updated if r["id"] == updated["id"] else r for r in records]
    if not any(r["id"] == updated["id"] for r in records):
        records.append(updated)
    records.sort(key=lambda r: r.get("received_at", ""), reverse=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in reversed(records)) + "\n"
    QUEUE_PATH.write_text(body, encoding="utf-8")
