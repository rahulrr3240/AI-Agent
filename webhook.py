"""Local alert webhook: POST a real alert into the live queue.

This is the machine-facing half of the live-alert path. Point an Alertmanager or
Grafana contact point at it (or use `scripts/send_alert.py`), and the normalized
alert lands in `.live_alerts/queue.jsonl`, which the Streamlit app reads.

    python webhook.py                       # http://127.0.0.1:8081
    python -m uvicorn webhook:app --port 8081

    POST /alert    queue an alert        GET /alerts    queued alerts, newest first
    GET /health    what is configured

It binds to 127.0.0.1 by default and has no authentication: it only appends to a
local file and never executes anything. Set `WEBHOOK_HOST` if you deliberately
want to expose it, and put TLS and auth in front of it.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, HTTPException

from oncall.config import load_settings
from oncall.live import QUEUE_PATH, enqueue, load_queue, parse_webhook
from oncall.tools import SimulatedWorld

app = FastAPI(title="OnCall Memory alert webhook", version="1.0")

_settings = load_settings()
_world = SimulatedWorld(_settings.data_dir)


def _normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Alertmanager / Grafana bodies through parse_webhook; a plain alert passes through."""
    if isinstance(payload.get("alerts"), list) or any(
            key in payload for key in ("title", "message", "evalMatches", "state")):
        return parse_webhook(payload)
    if payload.get("service") or payload.get("alert"):
        return [{
            "service": str(payload.get("service") or ""),
            "alert": str(payload.get("alert") or "")[:600],
            "severity": str(payload.get("severity") or "P2"),
            "source": "manual",
            "deploys": [],
            "logs_text": str(payload.get("logs") or "")[:40_000],
        }]
    return parse_webhook(payload)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "queue": str(QUEUE_PATH),
        "queued_alerts": len(load_queue()),
        "memory_bank": _settings.hindsight_bank_id,
        "groq_configured": _settings.has_groq,
        "hindsight_configured": _settings.has_hindsight,
    }


@app.get("/alerts")
def alerts(limit: int = 50) -> dict[str, Any]:
    queued = load_queue(limit)
    return {"count": len(queued), "alerts": queued}


@app.post("/alert")
def receive_alert(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Accept an Alertmanager or Grafana webhook body and queue what it describes.

    Every alert in the batch is queued separately, so a grouped Alertmanager
    notification still becomes one live incident per firing alert.
    """
    try:
        families = _world.families()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500,
                            detail=f"data/ is missing ({exc}); run this from the repo root") from exc

    records = []
    for alert in _normalize(payload):
        if not (alert.get("alert") or alert.get("service")):
            continue
        alert.setdefault("origin", "webhook /alert")
        record = enqueue(alert, families)
        records.append({
            "id": record["id"],
            "service": record["service"],
            "family": record["family"],
            "severity": record["severity"],
            "alert": record["alert"][:200],
            "has_logs": bool(record["logs_text"]),
        })

    if not records:
        raise HTTPException(status_code=422, detail=(
            "No alert found in the body. Expected an Alertmanager body with `alerts[]`, "
            "a Grafana body with `title`/`message`, or {\"service\", \"alert\"}."))

    return {"queued": len(records), "alerts": records, "queue_file": str(QUEUE_PATH),
            "next": "open the app: Live alert tab"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("WEBHOOK_HOST", "127.0.0.1"),
                port=int(os.getenv("WEBHOOK_PORT", "8081")))
