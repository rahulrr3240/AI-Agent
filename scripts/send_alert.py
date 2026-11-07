"""Post a sample alert to the local webhook so the live path can be exercised.

    python webhook.py                      # in another terminal (or --url elsewhere)
    python scripts/send_alert.py           # Alertmanager-shaped body
    python scripts/send_alert.py --kind grafana
    python scripts/send_alert.py --service orders-db --alert "disk pressure on orders-db"
    python scripts/send_alert.py --log-file ./tail-of-my.log

Then open the app's **Live alert** tab (or `GET /alerts`) to see it queued.
The sample payloads below look like real notifications, but they are examples:
nothing is executed and no real infrastructure is contacted.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

ALERTMANAGER_SAMPLE = {
    "version": "4",
    "status": "firing",
    "receiver": "oncall-memory-webhook",
    "groupLabels": {"alertname": "CheckoutTLSCertificateExpiring"},
    "alerts": [
        {
            "status": "firing",
            "labels": {
                "alertname": "CheckoutTLSCertificateExpiring",
                "service": "checkout-service",
                "severity": "critical",
                "cluster": "prod-eu-1",
            },
            "annotations": {
                "summary": "Upstream TLS handshakes are failing",
                "description": (
                    "checkout-service cannot complete the TLS handshake with "
                    "api.payments-gateway.com; x509: certificate has expired."
                ),
            },
            "startsAt": "",
            "fingerprint": "sample0001",
        }
    ],
}

GRAFANA_SAMPLE = {
    "title": "[FIRING:1] Checkout p95 latency (checkout-service prod-eu-1)",
    "state": "alerting",
    "message": "p95 latency is above the 2000ms SLO for 5m. ConnectionPoolTimeoutError in logs.",
    "startedAt": "",
    "tags": {"service": "checkout-service", "severity": "critical"},
    "evalMatches": [{"metric": "p95_latency_ms", "value": 4100}],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8081/alert")
    parser.add_argument("--kind", choices=["alertmanager", "grafana"], default="alertmanager")
    parser.add_argument("--service", default=None, help="override the sample's service")
    parser.add_argument("--alert", default=None, help="override the sample's alert text")
    parser.add_argument("--log-file", default=None, help="append these log lines to the payload")
    parser.add_argument("--severity", default=None, help="critical|warning|info (grafana body)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    payload = json.loads(json.dumps(ALERTMANAGER_SAMPLE if args.kind == "alertmanager"
                                    else GRAFANA_SAMPLE))
    stamp = now.isoformat().replace("+00:00", "Z")
    if args.kind == "alertmanager":
        alert = payload["alerts"][0]
        alert["startsAt"] = stamp
        if args.service:
            alert["labels"]["service"] = args.service
        if args.alert:
            alert["annotations"]["description"] = args.alert
    else:
        payload["startedAt"] = stamp
        if args.service:
            payload["tags"]["service"] = args.service
        if args.alert:
            payload["message"] = args.alert
        if args.severity:
            payload["severity"] = args.severity

    if args.log_file:
        text = open(args.log_file, encoding="utf-8", errors="replace").read()
        payload["logs"] = text[-40_000:]

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(args.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    print(f"POST {args.url}  ({args.kind}, {len(body)} bytes)")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"FAIL  HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}")
        return 1
    except urllib.error.URLError as exc:
        print(f"FAIL  cannot reach {args.url}: {exc.reason}")
        print("      start the receiver first:  python webhook.py")
        return 1

    print(json.dumps(result, indent=2))
    ids = ", ".join(a["id"] for a in result.get("alerts", []))
    print(f"\nQueued {result.get('queued')} live alert(s): {ids}")
    print("Next: streamlit run app.py -> Live alert tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
