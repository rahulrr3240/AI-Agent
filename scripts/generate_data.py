"""Generate the simulated production world, deterministically.

Everything the agent can read is produced here from hand-written family
templates: 28 resolved history incidents, 20 unresolved eval incidents,
matching deploy records, log files and metric snapshots. The bench and the UI
both read the generated JSON, so a story only has to be consistent once.

Run:  python scripts/generate_data.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oncall.schemas import EvalIncident, Incident  # noqa: E402

DATA = ROOT / "data"
LOGS = DATA / "logs"
METRICS = DATA / "metrics"

# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------
SERVICES = [
    {
        "name": "checkout-service",
        "team": "Payments & Checkout",
        "runtime": "Node.js 20",
        "dependencies": ["orders-db", "redis-cache", "payments-service"],
        "slo_p95_ms": 400,
        "slo_error_rate_pct": 0.5,
        "description": "Cart validation and order placement API.",
        "primary_oncall": "priya",
    },
    {
        "name": "auth-service",
        "team": "Identity",
        "runtime": "Go 1.22",
        "dependencies": ["postgres-identity", "redis-cache", "sso.internal"],
        "slo_p95_ms": 250,
        "slo_error_rate_pct": 0.2,
        "description": "Session issuance, token validation, OIDC federation.",
        "primary_oncall": "marcus",
    },
    {
        "name": "search-service",
        "team": "Discovery",
        "runtime": "Java 21",
        "dependencies": ["elasticsearch", "kafka"],
        "slo_p95_ms": 600,
        "slo_error_rate_pct": 1.0,
        "description": "Product search and ranking API.",
        "primary_oncall": "alice",
    },
    {
        "name": "payments-service",
        "team": "Payments & Checkout",
        "runtime": "Python 3.11 / FastAPI",
        "dependencies": ["orders-db", "payments-gateway", "kafka"],
        "slo_p95_ms": 800,
        "slo_error_rate_pct": 0.5,
        "description": "Charges, refunds and gateway reconciliation.",
        "primary_oncall": "dana",
    },
    {
        "name": "orders-db",
        "team": "Data Platform",
        "runtime": "PostgreSQL 16",
        "dependencies": ["s3-backups"],
        "slo_p95_ms": 120,
        "slo_error_rate_pct": 0.1,
        "description": "Primary orders database (primary + one replica).",
        "primary_oncall": "ravi",
    },
]
SERVICE_NAMES = {s["name"] for s in SERVICES}
DEPENDENCIES = {s["name"]: s["dependencies"] for s in SERVICES}
RUNTIME = {s["name"]: s["runtime"] for s in SERVICES}

# --------------------------------------------------------------------------
# Incident families
# --------------------------------------------------------------------------
FAMILIES = {
    "redis_pool_exhaustion": {
        "label": "Redis connection pool exhaustion after deploy",
        "signature": "ConnectionPoolTimeoutError",
        "keywords": ["connectionpooltimeouterror", "redis pool", "waiting for connection from pool", "redis"],
        "common_fixes": [
            "rollback_deploy_and_increase_pool",
            "increase_pool_size",
            "rollback_deploy",
        ],
        "generic_fixes": ["restart_service", "rollback_deploy", "scale_up"],
    },
    "memory_leak": {
        "label": "Memory leak in a new release",
        "signature": "OOMKilled",
        "keywords": ["oomkilled", "rss_growth", "memory limit", "memory climbing", "exceeded its memory limit"],
        "common_fixes": ["rollback_deploy"],
        "generic_fixes": ["restart_service", "scale_up", "rollback_deploy"],
    },
    "expired_tls_cert": {
        "label": "Expired TLS certificate",
        "signature": "certificate has expired",
        "keywords": ["certificate has expired", "x509", "certificate verify failed", "tls handshake", "failed to verify certificate"],
        "common_fixes": ["renew_certificate"],
        "generic_fixes": ["restart_service", "rollback_deploy", "scale_up"],
    },
    "slow_query_migration": {
        "label": "Slow query after a DB migration",
        "signature": "canceling statement due to lock timeout",
        "keywords": ["canceling statement due to lock timeout", "lock timeout", "seq scan", "slow query", "applied migration"],
        "common_fixes": ["add_db_index"],
        "generic_fixes": ["restart_service", "scale_up", "rollback_deploy"],
    },
    "payments_rate_limit": {
        "label": "Third-party API rate limiting",
        "signature": "429 Too Many Requests",
        "keywords": ["429 too many requests", "429", "retry-after", "rate_limit_remaining", "retry storm"],
        "common_fixes": ["enable_backoff_circuit_breaker"],
        "generic_fixes": ["restart_service", "scale_up", "rollback_deploy"],
    },
    "kafka_consumer_lag": {
        "label": "Kafka consumer lag (no history)",
        "signature": "consumer group lag",
        "keywords": ["consumer group lag", "consumer lag", "kafka", "poll timeout", "rebalance"],
        "common_fixes": ["scale_up"],
        "generic_fixes": ["restart_service", "scale_up", "clear_cache"],
    },
    "signing_key_rotation": {
        "label": "Signing-key rotation broke token validation (no history)",
        "signature": "invalid_client",
        "keywords": ["invalid_client", "signing key", "jwks", "jwt verify", "unknown kid"],
        "common_fixes": ["rollback_secret_rotation"],
        "generic_fixes": ["restart_service", "rollback_deploy", "clear_cache"],
    },
}

# --------------------------------------------------------------------------
# History: 28 resolved incidents, Mar-Oct 2026
# --------------------------------------------------------------------------
HISTORY_SPECS = [
    # -- Redis pool exhaustion ------------------------------------------------
    dict(id="INC-002", family="redis_pool_exhaustion", service="checkout-service",
         opened_at="2026-03-04T02:14:00Z", severity="P1", resolved_by="priya",
         resolution_minutes=25, outcome="rollback_deploy_and_increase_pool",
         version="v2.18.0", previous="v2.17.4", pool_old=100, pool_new=40),
    dict(id="INC-005", family="redis_pool_exhaustion", service="auth-service",
         opened_at="2026-03-19T09:41:00Z", severity="P2", resolved_by="marcus",
         resolution_minutes=31, outcome="rollback_deploy_and_increase_pool",
         version="v4.2.0", previous="v4.1.6", pool_old=80, pool_new=30),
    dict(id="INC-009", family="redis_pool_exhaustion", service="payments-service",
         opened_at="2026-04-11T18:02:00Z", severity="P2", resolved_by="dana",
         resolution_minutes=18, outcome="increase_pool_size",
         version=None, pool_old=20, pool_new=60, cause="config_drift"),
    dict(id="INC-014", family="redis_pool_exhaustion", service="checkout-service",
         opened_at="2026-06-02T03:37:00Z", severity="P1", resolved_by="priya",
         resolution_minutes=25, outcome="rollback_deploy_and_increase_pool",
         version="v2.31.0", previous="v2.30.2", pool_old=50, pool_new=20),
    dict(id="INC-019", family="redis_pool_exhaustion", service="auth-service",
         opened_at="2026-07-23T14:18:00Z", severity="P2", resolved_by="marcus",
         resolution_minutes=40, outcome="rollback_deploy",
         version="v4.7.1", previous="v4.7.0", pool_old=80, pool_new=25),
    dict(id="INC-024", family="redis_pool_exhaustion", service="checkout-service",
         opened_at="2026-09-03T02:14:00Z", severity="P1", resolved_by="priya",
         resolution_minutes=22, outcome="rollback_deploy_and_increase_pool",
         version="v2.40.0", previous="v2.39.3", pool_old=60, pool_new=25),
    # -- Memory leak ----------------------------------------------------------
    dict(id="INC-003", family="memory_leak", service="search-service",
         opened_at="2026-03-08T11:05:00Z", severity="P2", resolved_by="alice",
         resolution_minutes=35, outcome="rollback_deploy",
         version="v5.2.0", previous="v5.1.7", leak_rate=42),
    dict(id="INC-007", family="memory_leak", service="payments-service",
         opened_at="2026-03-27T06:52:00Z", severity="P2", resolved_by="dana",
         resolution_minutes=28, outcome="rollback_deploy",
         version="v2.0.0", previous="v1.9.8", leak_rate=55),
    dict(id="INC-012", family="memory_leak", service="checkout-service",
         opened_at="2026-05-06T21:13:00Z", severity="P1", resolved_by="priya",
         resolution_minutes=33, outcome="rollback_deploy",
         version="v2.25.1", previous="v2.24.9", leak_rate=63),
    dict(id="INC-017", family="memory_leak", service="search-service",
         opened_at="2026-07-02T08:29:00Z", severity="P2", resolved_by="alice",
         resolution_minutes=41, outcome="rollback_deploy",
         version="v5.9.0", previous="v5.8.4", leak_rate=48),
    dict(id="INC-021", family="memory_leak", service="search-service",
         opened_at="2026-08-14T19:44:00Z", severity="P2", resolved_by="alice",
         resolution_minutes=38, outcome="rollback_deploy",
         version="v6.1.0", previous="v6.0.3", leak_rate=71),
    dict(id="INC-026", family="memory_leak", service="payments-service",
         opened_at="2026-09-24T04:06:00Z", severity="P2", resolved_by="dana",
         resolution_minutes=26, outcome="rollback_deploy",
         version="v2.14.0", previous="v2.13.6", leak_rate=39),
    # -- Expired TLS certificate ---------------------------------------------
    dict(id="INC-001", family="expired_tls_cert", service="payments-service",
         opened_at="2026-03-01T02:12:00Z", severity="P1", resolved_by="dana",
         resolution_minutes=47, cert_cn="api.payments-gateway.com", cert_expired="2026-02-28T23:59:59Z"),
    dict(id="INC-008", family="expired_tls_cert", service="checkout-service",
         opened_at="2026-04-02T07:31:00Z", severity="P1", resolved_by="priya",
         resolution_minutes=52, cert_cn="api.payments-gateway.com", cert_expired="2026-04-01T00:00:00Z"),
    dict(id="INC-013", family="expired_tls_cert", service="auth-service",
         opened_at="2026-05-27T16:09:00Z", severity="P1", resolved_by="marcus",
         resolution_minutes=39, cert_cn="sso.internal", cert_expired="2026-05-27T00:00:00Z"),
    dict(id="INC-020", family="expired_tls_cert", service="payments-service",
         opened_at="2026-08-03T03:48:00Z", severity="P1", resolved_by="dana",
         resolution_minutes=44, cert_cn="api.payments-gateway.com", cert_expired="2026-08-02T12:00:00Z"),
    dict(id="INC-027", family="expired_tls_cert", service="checkout-service",
         opened_at="2026-10-05T22:26:00Z", severity="P2", resolved_by="priya",
         resolution_minutes=36, cert_cn="api.payments-gateway.com", cert_expired="2026-10-05T18:00:00Z"),
    # -- Slow query after migration ------------------------------------------
    dict(id="INC-004", family="slow_query_migration", service="orders-db",
         opened_at="2026-03-12T13:22:00Z", severity="P1", resolved_by="ravi",
         resolution_minutes=62, migration="0042_add_orders_region", table="orders",
         column="customer_id, created_at"),
    dict(id="INC-010", family="slow_query_migration", service="orders-db",
         opened_at="2026-04-17T10:04:00Z", severity="P1", resolved_by="ravi",
         resolution_minutes=55, migration="0051_add_orders_channel", table="orders",
         column="status, updated_at"),
    dict(id="INC-015", family="slow_query_migration", service="checkout-service",
         opened_at="2026-06-11T15:40:00Z", severity="P2", resolved_by="priya",
         resolution_minutes=48, migration="0058_orders_promo_code", table="orders",
         column="promo_code, created_at", deploy_service="orders-db"),
    dict(id="INC-018", family="slow_query_migration", service="orders-db",
         opened_at="2026-07-15T01:57:00Z", severity="P1", resolved_by="ravi",
         resolution_minutes=58, migration="0063_orders_fulfilment_state", table="orders",
         column="fulfilment_state"),
    dict(id="INC-022", family="slow_query_migration", service="payments-service",
         opened_at="2026-08-21T12:33:00Z", severity="P2", resolved_by="dana",
         resolution_minutes=51, migration="0069_orders_settlement_id", table="orders",
         column="settlement_id", deploy_service="orders-db"),
    dict(id="INC-025", family="slow_query_migration", service="orders-db",
         opened_at="2026-09-12T05:19:00Z", severity="P1", resolved_by="ravi",
         resolution_minutes=47, migration="0074_orders_refund_state", table="orders",
         column="refund_state, created_at"),
    # -- Third-party rate limiting -------------------------------------------
    dict(id="INC-006", family="payments_rate_limit", service="payments-service",
         opened_at="2026-03-22T20:15:00Z", severity="P2", resolved_by="dana",
         resolution_minutes=30, gateway="payments-gateway", retry_after=12),
    dict(id="INC-011", family="payments_rate_limit", service="checkout-service",
         opened_at="2026-04-25T11:08:00Z", severity="P2", resolved_by="priya",
         resolution_minutes=34, gateway="payments-gateway", retry_after=20),
    dict(id="INC-016", family="payments_rate_limit", service="payments-service",
         opened_at="2026-06-19T17:51:00Z", severity="P2", resolved_by="dana",
         resolution_minutes=27, gateway="payments-gateway", retry_after=8),
    dict(id="INC-023", family="payments_rate_limit", service="checkout-service",
         opened_at="2026-08-28T09:37:00Z", severity="P2", resolved_by="priya",
         resolution_minutes=32, gateway="payments-gateway", retry_after=15),
    dict(id="INC-028", family="payments_rate_limit", service="payments-service",
         opened_at="2026-10-14T14:03:00Z", severity="P1", resolved_by="dana",
         resolution_minutes=29, gateway="payments-gateway", retry_after=25),
]

# --------------------------------------------------------------------------
# Eval: 20 unresolved incidents (18 recurring + 2 novel), never seeded
# --------------------------------------------------------------------------
EVAL_SPECS = [
    dict(id="INC-101", family="redis_pool_exhaustion", service="checkout-service",
         opened_at="2026-10-20T02:09:00Z", true_fix="rollback_deploy_and_increase_pool",
         version="v2.44.1", previous="v2.44.0", pool_old=60, pool_new=20),
    dict(id="INC-102", family="memory_leak", service="search-service",
         opened_at="2026-10-22T08:14:00Z", true_fix="rollback_deploy",
         version="v6.4.0", previous="v6.3.2", leak_rate=58),
    dict(id="INC-103", family="expired_tls_cert", service="payments-service",
         opened_at="2026-10-24T04:52:00Z", true_fix="renew_certificate",
         cert_cn="api.payments-gateway.com", cert_expired="2026-10-23T12:00:00Z"),
    dict(id="INC-104", family="slow_query_migration", service="orders-db",
         opened_at="2026-10-26T11:26:00Z", true_fix="add_db_index",
         migration="0079_orders_loyalty_tier", table="orders", column="loyalty_tier"),
    dict(id="INC-105", family="payments_rate_limit", service="payments-service",
         opened_at="2026-10-28T19:38:00Z", true_fix="enable_backoff_circuit_breaker",
         gateway="payments-gateway", retry_after=18),
    dict(id="INC-106", family="redis_pool_exhaustion", service="auth-service",
         opened_at="2026-10-30T06:03:00Z", true_fix="increase_pool_size",
         version=None, pool_old=20, pool_new=80, cause="config_drift"),
    dict(id="INC-107", family="memory_leak", service="payments-service",
         opened_at="2026-11-01T13:44:00Z", true_fix="rollback_deploy",
         version="v2.19.0", previous="v2.18.4", leak_rate=64),
    dict(id="INC-108", family="expired_tls_cert", service="checkout-service",
         opened_at="2026-11-02T01:17:00Z", true_fix="renew_certificate",
         cert_cn="api.payments-gateway.com", cert_expired="2026-11-01T06:00:00Z"),
    dict(id="INC-109", family="slow_query_migration", service="orders-db",
         opened_at="2026-11-03T03:28:00Z", true_fix="add_db_index",
         migration="0082_orders_gift_message", table="orders", column="gift_message_id"),
    dict(id="INC-110", family="payments_rate_limit", service="checkout-service",
         opened_at="2026-11-03T15:52:00Z", true_fix="enable_backoff_circuit_breaker",
         gateway="payments-gateway", retry_after=10),
    dict(id="INC-111", family="redis_pool_exhaustion", service="checkout-service",
         opened_at="2026-11-04T02:41:00Z", true_fix="rollback_deploy_and_increase_pool",
         version="v2.46.0", previous="v2.45.2", pool_old=60, pool_new=25),
    dict(id="INC-112", family="memory_leak", service="search-service",
         opened_at="2026-11-04T09:06:00Z", true_fix="rollback_deploy",
         version="v6.6.1", previous="v6.6.0", leak_rate=47),
    dict(id="INC-113", family="slow_query_migration", service="payments-service",
         opened_at="2026-11-05T00:34:00Z", true_fix="add_db_index",
         migration="0084_orders_dispute_flag", table="orders", column="dispute_flag",
         deploy_service="orders-db"),
    dict(id="INC-114", family="payments_rate_limit", service="payments-service",
         opened_at="2026-11-05T12:19:00Z", true_fix="enable_backoff_circuit_breaker",
         gateway="payments-gateway", retry_after=22),
    dict(id="INC-115", family="expired_tls_cert", service="auth-service",
         opened_at="2026-11-06T00:07:00Z", true_fix="renew_certificate",
         cert_cn="sso.internal", cert_expired="2026-11-05T20:00:00Z"),
    dict(id="INC-116", family="redis_pool_exhaustion", service="payments-service",
         opened_at="2026-11-06T02:22:00Z", true_fix="rollback_deploy_and_increase_pool",
         version="v2.21.0", previous="v2.20.3", pool_old=60, pool_new=20),
    dict(id="INC-117", family="memory_leak", service="checkout-service",
         opened_at="2026-11-06T04:11:00Z", true_fix="rollback_deploy",
         version="v2.47.0", previous="v2.46.1", leak_rate=69),
    dict(id="INC-119", family="kafka_consumer_lag", service="search-service",
         opened_at="2026-11-06T06:02:00Z", true_fix="scale_up", novel=True,
         lag=184320),
    dict(id="INC-120", family="signing_key_rotation", service="auth-service",
         opened_at="2026-11-06T06:41:00Z", true_fix="rollback_secret_rotation", novel=True,
         kid="2026-10-31"),
    # Second occurrence of the kafka pattern: condition C can recall the outcome
    # it learned at INC-119; condition B still has no history for it.
    dict(id="INC-121", family="kafka_consumer_lag", service="search-service",
         opened_at="2026-11-06T07:14:00Z", true_fix="scale_up",
         lag=221880),
]

# Unrelated deploys (red herrings + background noise) -----------------------
NOISE_DEPLOYS = [
    ("checkout-service", "v2.19.0", "2026-03-11T15:20:00Z", "priya", "Tighten cart validation error codes", "9c1e4a2", "v2.18.3"),
    ("checkout-service", "v2.32.0", "2026-06-15T10:05:00Z", "priya", "Add gift-message field to cart", "b77d310", "v2.31.7"),
    ("checkout-service", "v2.41.0", "2026-09-10T14:30:00Z", "priya", "Retry idempotency keys on 5xx", "4f2ab8e", "v2.40.3"),
    ("checkout-service", "v2.44.0", "2026-10-19T09:12:00Z", "priya", "Reduce Redis client keepalive to 30s", "e91c0f7", "v2.43.6"),
    ("checkout-service", "v2.44.2", "2026-10-21T11:41:00Z", "priya", "Fix nil guard in promo resolver", "5bd8021", "v2.44.1"),
    ("auth-service", "v4.3.0", "2026-04-02T12:10:00Z", "marcus", "Add PKCE support to OIDC client", "c3ea118", "v4.2.4"),
    ("auth-service", "v4.7.0", "2026-07-21T18:22:00Z", "marcus", "Session store: switch to pipelined writes", "1a9f6b3", "v4.6.2"),
    ("auth-service", "v4.9.0", "2026-10-29T08:47:00Z", "marcus", "Rotate OIDC signing key (kid=2026-10-31)", "d5b2077", "v4.8.1"),
    ("auth-service", "v4.9.1", "2026-11-05T22:38:00Z", "marcus", "Rotate OIDC signing key (kid=2026-11-05)", "ee4c9a0", "v4.9.0"),
    ("search-service", "v5.10.1", "2026-04-09T09:31:00Z", "alice", "Re-rank: lower synonym weight", "8ab144f", "v5.10.0"),
    ("search-service", "v6.2.0", "2026-08-25T13:03:00Z", "alice", "Index product facets as keyword fields", "0c71d5e", "v6.1.4"),
    ("search-service", "v6.5.9", "2026-10-26T08:12:00Z", "alice", "Cache embedding model per shard", "3de9a81", "v6.5.6"),
    ("search-service", "v6.4.1", "2026-10-23T10:26:00Z", "alice", "Bump Lucene to 9.11.1", "6aa02c4", "v6.4.0"),
    ("payments-service", "v2.1.0", "2026-04-05T11:55:00Z", "dana", "Add idempotency key to refunds", "f40cc21", "v2.0.6"),
    ("payments-service", "v2.10.0", "2026-07-08T16:14:00Z", "dana", "Gateway client: reuse HTTP connection pool", "b2d7710", "v2.9.4"),
    ("payments-service", "v2.17.0", "2026-10-08T07:48:00Z", "dana", "Settlement reconciliation job", "91ce6d3", "v2.16.2"),
    ("payments-service", "v2.19.0", "2026-10-29T12:58:00Z", "dana", "Upgrade protobuf to 5.28", "c8a3f52", "v2.18.4"),
    ("orders-db", "migration-0077", "2026-10-11T02:00:00Z", "ravi", "Add orders.promo_source column", "d00a117", "migration-0076"),
    ("orders-db", "migration-0080", "2026-10-27T02:00:00Z", "ravi", "Backfill orders.loyalty_tier", "fa8210c", "migration-0079"),
    ("orders-db", "migration-0085", "2026-11-05T02:00:00Z", "ravi", "Add orders.checkout_session_id", "0bb19d4", "migration-0084"),
    ("orders-db", "migration-0087", "2026-11-06T02:20:00Z", "ravi", "Backfill checkout_session_id in batches", "5e7c230", "migration-0086"),
]

GENERIC_FIXES_FAILED = {
    "restart_service": [
        "pods restarted clean, but the alert fired again after {n} min",
        "restart gave {n} min of relief, then the same errors returned",
        "rolling restart completed; no lasting improvement",
    ],
    "rollback_deploy": [
        "rolled back one release; errors unchanged, so the deploy was not the cause",
        "rollback completed but the failure persisted",
    ],
    "scale_up": [
        "added 4 replicas; throughput flat, latency unchanged",
        "scaled out, but the bottleneck moved rather than disappeared",
    ],
    "increase_pool_size": [
        "raised pool to {pool} but connections were still held by blocking calls",
        "pool increase applied; waiting connections did not drop",
    ],
    "clear_cache": [
        "cache flushed; error rate unchanged and latency briefly worse",
    ],
}

WORKED_NOTES = {
    "rollback_deploy_and_increase_pool": "rolled back to {previous} and raised the pool to {pool}",
    "rollback_deploy": "rolled back to {previous}; memory and OOM restarts returned to baseline",
    "increase_pool_size": "raised pool to {pool}; waiting connections dropped to zero",
    "renew_certificate": "renewed {cert_cn} and reloaded it on all pods; handshakes recovered",
    "add_db_index": "created index on {table}({column}); query dropped from 8.4s to 41ms",
    "enable_backoff_circuit_breaker": "enabled exponential backoff + circuit breaker; 429s stopped cascading",
    "restart_service": "restart cleared the condition",
    "scale_up": "scaled the consumer deployment from 6 to 18 pods; lag drained in 9 min",
    "clear_cache": "cache cleared and repopulated",
    "rollback_secret_rotation": "restored the previous signing key and re-published the JWKS",
}


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def deploy_plan(incident_id: str, spec: dict, opened: datetime) -> list[dict]:
    """Deploy record triggered by / associated with an incident, if any."""
    version = spec.get("version")
    if spec.get("migration"):
        version = f"migration-{spec['migration'][:4]}"
    if not version:
        return []
    service = spec.get("deploy_service", spec["service"])
    offset = spec.get("deploy_offset_min", 45)
    return [{
        "service": service,
        "version": version,
        "deployed_at": iso(opened - timedelta(minutes=offset)),
        "deployed_by": {"orders-db": "ravi"}.get(service, spec.get("resolved_by", "priya")),
        "summary": _deploy_summary(spec),
        "commit": _commit_for(incident_id),
        "previous_version": spec.get("previous"),
        "incident_id": incident_id,
    }]


def _target_pool(spec: dict) -> int:
    """Pool size the engineer raised the pool to (never a reduction)."""
    return max(spec.get("pool_old") or 0, spec.get("pool_new") or 0, 60)


def _deploy_summary(spec: dict) -> str:
    family = spec["family"]
    if family == "redis_pool_exhaustion":
        return f"Reduce Redis pool default from {spec.get('pool_old')} to {spec.get('pool_new')} connections"
    if family == "memory_leak":
        return "Rewrite response serialisation path; add in-memory response cache"
    if family == "slow_query_migration":
        return f"Migration {spec.get('migration')}: add column and backfill {spec.get('table')}"
    return "Routine release"


def _commit_for(seed: str) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("0123456789abcdef") for _ in range(7))


def make_logs(inc: dict) -> str:
    """Build a service-appropriate log file for the incident story."""
    family = inc["family"]
    service = inc["service"]
    runtime = RUNTIME[service]
    opened = utc(inc["opened_at"])
    rng = random.Random(inc["id"] + family)
    lines: list[tuple[datetime, str, str]] = []

    def add(minute_offset: float, level: str, msg: str) -> None:
        lines.append((opened + timedelta(minutes=minute_offset), level, msg))

    def fmt(dt: datetime, level: str, msg: str) -> str:
        ms = dt.microsecond // 1000
        if runtime.startswith("Node"):
            return f"{dt:%Y-%m-%dT%H:%M:%S}.{ms:03d}Z {level:<5} {service} {msg}"
        if runtime.startswith("Go"):
            return f'time={dt:%Y-%m-%dT%H:%M:%SZ} level={level.lower()} service={service} msg="{msg}"'
        if runtime.startswith("Java"):
            return (f"{dt:%Y-%m-%d %H:%M:%S}.{ms:03d} {level:<5} [ {service} ] "
                    f"[http-nio-8080-exec-{rng.randint(2, 9)}] c.v.search.SearchController - {msg}")
        if runtime.startswith("PostgreSQL"):
            return f"{dt:%Y-%m-%d %H:%M:%S}.{ms:03d} UTC [{rng.randint(14000, 19999)}] {level}:  {msg}"
        return f"{dt:%Y-%m-%d %H:%M:%S},{ms:03d} {level:<7} [{service}] {msg}"

    version = inc.get("version")
    if version:
        add(-6.4, "INFO", f"starting {service} version {version} (commit {_commit_for(inc['id'])})")
    if family == "redis_pool_exhaustion":
        pool = inc.get("pool_size", inc.get("pool_new", 20))
        add(-6.1, "INFO", f"redis pool initialised size={pool} min_idle=2 timeout=2000ms")
        add(-3.4, "WARN", f"redis pool exhausted active={pool} idle=0 waiting={rng.randint(14, 37)}")
        add(-2.7, "ERROR", "ConnectionPoolTimeoutError: Timeout waiting for connection from pool (2000ms)")
        if runtime.startswith("Node"):
            add(-2.2, "ERROR", f"GET /api/checkout 504 4123ms request_id=req_{rng.randint(10000,99999)}")
            add(-1.1, "ERROR", "ConnectionPoolTimeoutError: Timeout waiting for connection from pool (2000ms)")
            add(0.4, "WARN", "redis pool saturated: cmd=GET key=cart:{session} queue_depth=41")
        elif runtime.startswith("Go"):
            add(-2.0, "ERROR", f"redis: ConnectionPoolTimeoutError: timeout waiting for connection from pool pool_size={pool}")
            add(-0.9, "ERROR", f"POST /v1/session/refresh 503 latency=2.001s trace={_commit_for(inc['id'] + 't')}")
        else:
            add(-2.4, "ERROR", "ConnectionPoolTimeoutError: Timeout waiting for connection from pool")
            add(-1.3, "ERROR", f"POST /v1/charges 503 provider=redis-cache elapsed=2.01s")
            add(0.2, "WARN", "retry 1/2 after redis timeout")
    elif family == "memory_leak":
        limit = {"checkout-service": 2048, "search-service": 4096, "payments-service": 1536}.get(service, 2048)
        add(-5.6, "WARN", f"container memory {int(limit*0.92)}MiB / limit {limit}MiB rss_growth=+{inc.get('leak_rate', 40)}MiB/min")
        add(-4.2, "ERROR", f"kubelet: container {service} restarted: OOMKilled (exit code 137)")
        add(-3.3, "INFO", f"probe /healthz ok after restart; memory now {int(limit*0.35)}MiB")
        add(-1.8, "WARN", f"memory climbing again: {int(limit*0.78)}MiB rss_growth=+{inc.get('leak_rate', 40)}MiB/min")
        add(-0.7, "ERROR", "OOMKilled: container exceeded its memory limit")
        add(0.9, "ERROR", f"kubelet: BackOff restarting container {service}")
    elif family == "expired_tls_cert":
        cert_cn = inc.get("cert_cn", "api.payments-gateway.com")
        expired = inc.get("cert_expired", "2026-01-01T00:00:00Z")
        add(-4.8, "ERROR", f"upstream handshake failed: x509: certificate has expired or is not yet valid: certificate for {cert_cn} expired at {expired}")
        add(-4.1, "WARN", f"tls: failed to verify certificate: x509: certificate has expired ({cert_cn})")
        if runtime.startswith("Node"):
            add(-3.2, "ERROR", f"POST https://{cert_cn}/v1/charges failed: certificate has expired code=DEPTH_ZERO_SELF_SIGNED_CERT")
            add(-1.4, "ERROR", "circuit breaker opened for payments-gateway after 23 failures")
        elif runtime.startswith("Go"):
            add(-3.0, "ERROR", f"oidc discovery failed: Get https://{cert_cn}/.well-known/openid-configuration: tls: failed to verify certificate: x509: certificate has expired")
            add(-1.6, "ERROR", "token exchange aborted: upstream TLS validation error")
        else:
            add(-3.4, "ERROR", f"SSLError: certificate verify failed: certificate has expired (_ssl.c:1006) host={cert_cn}")
            add(-1.2, "ERROR", "gateway request aborted: TLS handshake failure")
    elif family == "slow_query_migration":
        table = inc.get("table", "orders")
        column = inc.get("column", "customer_id")
        migration = inc.get("migration", "0000")
        is_db = runtime.startswith("PostgreSQL")
        add(-5.9, "INFO", f"applied migration-{migration[:4]} ({migration}); backfill complete")
        if is_db:
            add(-4.6, "LOG", f"duration: 8421.331 ms  statement: SELECT id, status, total FROM {table} WHERE {column.split(',')[0]} = $1 ORDER BY created_at DESC LIMIT 20")
            add(-3.9, "LOG", f"process {rng.randint(18000,19000)} still waiting for ShareLock on transaction {rng.randint(80000,90000)} after 1000.219 ms")
            add(-3.1, "ERROR", "canceling statement due to lock timeout")
            add(-2.9, "STATEMENT", f"UPDATE {table} SET status = 'paid', updated_at = now() WHERE id = $1")
            add(-1.7, "LOG", "duration: 9012.774 ms  statement: SELECT count(*) FROM orders WHERE status = 'pending'")
            add(-1.2, "ERROR", "canceling statement due to lock timeout")
            add(0.6, "LOG", "checkpoint starting: time=0.4s")
        else:
            add(-3.6, "WARN", "db query slow: 5127ms (threshold 500ms) query=orders.by_customer")
            add(-3.0, "ERROR", "db error: canceling statement due to lock timeout (SQLSTATE 55P03)")
            add(-1.9, "ERROR", "upstream orders-db request failed after 5000ms: lock timeout")
            add(0.3, "WARN", "connection pool waiting=18 active=20")
    elif family == "payments_rate_limit":
        gateway = inc.get("gateway", "payments-gateway")
        retry_after = inc.get("retry_after", 12)
        add(-4.9, "WARN", f"gateway {gateway} responded 429 (retry-after: {retry_after}s) attempt=1/3")
        add(-4.3, "ERROR", f"charge failed: HTTP 429 Too Many Requests from {gateway} request_id=req_{rng.randint(10000,99999)}")
        add(-3.5, "WARN", f"gateway {gateway} responded 429 (retry-after: {retry_after + 4}s) attempt=2/3")
        add(-2.6, "ERROR", f"HTTP 429 Too Many Requests from {gateway} (rate_limit_remaining=0)")
        add(-1.8, "ERROR", "circuit breaker still closed: failures=17/20 window=60s")
        add(-0.8, "WARN", "retry storm: 143 in-flight retries, no jitter")
    elif family == "kafka_consumer_lag":
        lag = inc.get("lag", 184320)
        add(-5.2, "WARN", f"consumer group search-indexer lag={lag} partitions=12")
        add(-4.1, "WARN", "rebalance completed in 43s; 6 consumers assigned 12 partitions")
        add(-2.8, "ERROR", "KafkaConsumer poll timeout: no records for 30000ms")
        add(-1.4, "WARN", f"consumer group lag={lag + 15200} (growing)")
        add(0.7, "WARN", "index freshness SLO breached: newest indexed product is 41 min old")
    elif family == "signing_key_rotation":
        kid = inc.get("kid", "2026-10-31")
        add(-5.0, "INFO", f"JWKS refresh from sso.internal (keys=3)")
        add(-4.2, "ERROR", f"OIDC token validation failed: invalid_client: signing key kid={kid} not recognized")
        add(-3.1, "WARN", "jwt verify failed for issuer https://sso.internal (unknown kid)")
        add(-1.6, "ERROR", "401 rate above threshold: 62% of /v1/session/verify")
        add(0.5, "ERROR", f"token exchange rejected: invalid_client (kid={kid})")

    lines.sort(key=lambda x: x[0])
    header = [
        f"# simulated log extract — {inc['id']} — {service} ({runtime})",
        f"# window: {iso(opened - timedelta(minutes=7))} .. {iso(opened + timedelta(minutes=2))}",
    ]
    return "\n".join(header + [fmt(dt, lvl, msg) for dt, lvl, msg in lines]) + "\n"


def make_metrics(inc: dict) -> dict:
    family = inc["family"]
    service = inc["service"]
    opened = utc(inc["opened_at"])
    rng = random.Random(inc["id"] + "metrics")
    points = []
    base_p95 = {"checkout-service": 380, "auth-service": 240, "search-service": 560,
                "payments-service": 760, "orders-db": 110}.get(service, 400)
    for i in range(-6, 7):
        t = opened + timedelta(minutes=i)
        spike = 0 if i < 0 else max(0, 1 - abs(i - 1) / 5)
        spike_ms = base_p95 * 8 if family != "slow_query_migration" else 8200
        p95 = base_p95 + (spike * spike_ms) + rng.randint(-25, 25)
        err = 0.2 + (spike * 18) + rng.random() * 0.2
        point = {
            "t": iso(t),
            "p95_latency_ms": round(p95),
            "error_rate_pct": round(err, 2),
            "cpu_pct": round(min(99, 28 + spike * 55 + rng.random() * 6), 1),
        }
        if family == "redis_pool_exhaustion":
            point["redis_pool_waiting"] = int(max(0, spike * rng.randint(22, 44)))
            point["redis_pool_in_use"] = int(inc.get("pool_size", inc.get("pool_new", 20)) * (0.6 + spike * 0.4))
        elif family == "memory_leak":
            limit = {"checkout-service": 2048, "search-service": 4096, "payments-service": 1536}.get(service, 2048)
            point["memory_mb"] = int(min(limit * 1.02, limit * 0.5 + max(0, i + 6) * inc.get("leak_rate", 40) * 1.7))
            point["oom_restarts"] = 1 if i >= 2 else 0
        elif family == "expired_tls_cert":
            point["upstream_tls_failures"] = int(max(0, spike * rng.randint(40, 90)))
        elif family == "slow_query_migration":
            point["db_cpu_pct"] = round(min(99, 31 + spike * 60 + rng.random() * 5), 1)
            point["lock_waits"] = int(max(0, spike * rng.randint(18, 40)))
            point["seq_scans_per_min"] = int(max(0, spike * rng.randint(120, 240)))
        elif family == "payments_rate_limit":
            point["gateway_429s"] = int(max(0, spike * rng.randint(60, 140)))
        elif family == "kafka_consumer_lag":
            point["consumer_lag"] = int(max(0, inc.get("lag", 184320) + max(0, i + 6) * 6200))
        elif family == "signing_key_rotation":
            point["auth_401s"] = int(max(0, spike * rng.randint(300, 700)))
        points.append(point)
    summary = {
        "window_minutes": 12,
        "peak_p95_ms": max(p["p95_latency_ms"] for p in points),
        "peak_error_rate_pct": max(p["error_rate_pct"] for p in points),
        "service": service,
        "family": family,
    }
    return {"incident_id": inc["id"], "service": service, "summary": summary, "series": points}


def build_root_cause(spec: dict) -> str:
    f = spec["family"]
    s = spec["service"]
    if f == "redis_pool_exhaustion":
        if spec.get("cause") == "config_drift":
            return (f"Redis pool on {s} had been left at {spec.get('pool_old')} connections after a "
                    f"configmap edit; peak concurrency needs ~{spec.get('pool_new')}. No deploy involved.")
        return (f"Redis connection pool exhausted after deploy {spec.get('version')} cut the pool from "
                f"{spec.get('pool_old')} to {spec.get('pool_new')} while peak concurrency stayed around "
                f"{int(spec.get('pool_old', 60) * 1.15)}.")
    if f == "memory_leak":
        return (f"Leak in {spec.get('version')}: the new in-memory response cache is never evicted, RSS grows "
                f"~{spec.get('leak_rate')} MiB/min until the container hits its limit and is OOMKilled.")
    if f == "expired_tls_cert":
        return (f"The certificate for {spec.get('cert_cn')} expired at {spec.get('cert_expired')}; automatic "
                f"renewal was never wired up, so every outbound handshake fails verification.")
    if f == "slow_query_migration":
        return (f"Migration {spec.get('migration')} added a filter on {spec.get('table')}({spec.get('column')}) "
                f"without an index; the planner falls back to a sequential scan and row locks pile up.")
    if f == "payments_rate_limit":
        return (f"Retry traffic to {spec.get('gateway')} exceeded its 120 req/min limit after the client "
                f"started retrying immediately with no jitter, producing 429 storms.")
    if f == "kafka_consumer_lag":
        return (f"A product reindex backfill pushed {spec.get('lag')} messages into the search-indexer topic; "
                f"6 consumers cannot keep up with 12 partitions.")
    if f == "signing_key_rotation":
        return (f"SSO rotated its signing key to kid={spec.get('kid')} but auth-service's JWKS cache was not "
                f"invalidated, so tokens signed with the new key fail validation.")
    return "Unknown"


def build_symptoms(spec: dict) -> list[str]:
    f = spec["family"]
    s = spec["service"]
    p95 = {"checkout-service": "4.1s", "auth-service": "2.8s", "search-service": "5.2s",
           "payments-service": "3.9s", "orders-db": "6.4s"}.get(s, "4.0s")
    if f == "redis_pool_exhaustion":
        return [f"p95 latency {p95} (SLO 400ms)", "ConnectionPoolTimeoutError in logs",
                "Redis pool waiting queue > 20"]
    if f == "memory_leak":
        return ["pods OOMKilled and restarting", "RSS growing steadily since the release",
                f"memory near limit on {s}"]
    if f == "expired_tls_cert":
        return ["upstream TLS handshake failures", "x509 certificate has expired",
                "error rate above SLO for 5m"]
    if f == "slow_query_migration":
        return ["database CPU saturated", "lock waits climbing", "query timeouts from upstream services"]
    if f == "payments_rate_limit":
        return ["HTTP 429 from payments-gateway", "retry storm, no jitter", "charge failures"]
    if f == "kafka_consumer_lag":
        return [f"consumer group lag {spec.get('lag')}", "index freshness SLO breached"]
    if f == "signing_key_rotation":
        return ["62% of token validations return 401", "invalid_client from OIDC verifier"]
    return []


def build_alert(spec: dict) -> str:
    f = spec["family"]
    s = spec["service"]
    if f == "redis_pool_exhaustion":
        return f"{s} p95 latency > 2000ms for 5m"
    if f == "memory_leak":
        return f"{s} pods OOMKilling repeatedly (restart count > 2 in 15m)"
    if f == "expired_tls_cert":
        return f"{s} upstream error rate > 10% — TLS handshake failures"
    if f == "slow_query_migration":
        return f"{s} query latency > 5s and DB CPU > 90% for 5m"
    if f == "payments_rate_limit":
        return f"{s} error rate > 5% — HTTP 429 from payments-gateway"
    if f == "kafka_consumer_lag":
        return f"{s} consumer lag > 100k messages for 10m"
    if f == "signing_key_rotation":
        return f"{s} 401 rate > 25% for 5m"
    return f"{s} alert"


def build_fixes(spec: dict) -> list[dict]:
    """Compose the fixes tried, in the order the engineer tried them."""
    f = spec["family"]
    outcome = spec.get("outcome")
    rng = random.Random(spec["id"] + "fixes")
    fixes: list[dict] = []

    def fail(fix: str, extra: dict | None = None) -> dict:
        note = rng.choice(GENERIC_FIXES_FAILED.get(fix, ["no effect"]))
        note = note.format(n=6 + rng.randint(0, 8), pool=spec.get("pool_new", 60))
        return {"fix": fix, "outcome": "failed", "note": note, **(extra or {})}

    if f == "redis_pool_exhaustion":
        fixes.append(fail("restart_service"))
        if spec.get("cause") == "config_drift":
            fixes.append({"fix": "increase_pool_size", "outcome": "worked",
                          "note": WORKED_NOTES["increase_pool_size"].format(pool=_target_pool(spec))})
        elif spec.get("id") == "INC-019":
            fixes.append(fail("increase_pool_size"))
            fixes.append({"fix": "rollback_deploy", "outcome": "worked",
                          "note": f"rolled back to {spec.get('previous')}; the blocking call disappeared"})
        elif outcome:
            fixes.append({"fix": outcome, "outcome": "worked",
                          "note": WORKED_NOTES[outcome].format(previous=spec.get("previous"), pool=_target_pool(spec))})
    elif f == "memory_leak":
        fixes.append(fail("restart_service"))
        if spec.get("id") in {"INC-017", "INC-021"}:
            fixes.append(fail("scale_up"))
        if outcome:
            fixes.append({"fix": outcome, "outcome": "worked",
                          "note": WORKED_NOTES[outcome].format(previous=spec.get("previous"))})
    elif f == "expired_tls_cert":
        fixes.append(fail("restart_service"))
        fixes.append(fail("rollback_deploy"))
        fixes.append({"fix": "renew_certificate", "outcome": "worked",
                      "note": WORKED_NOTES["renew_certificate"].format(cert_cn=spec.get("cert_cn"))})
    elif f == "slow_query_migration":
        fixes.append(fail("restart_service"))
        fixes.append(fail("scale_up"))
        fixes.append({"fix": "add_db_index", "outcome": "worked",
                      "note": WORKED_NOTES["add_db_index"].format(table=spec.get("table"), column=spec.get("column"))})
    elif f == "payments_rate_limit":
        fixes.append(fail("restart_service"))
        fixes.append({"fix": "enable_backoff_circuit_breaker", "outcome": "worked",
                      "note": WORKED_NOTES["enable_backoff_circuit_breaker"]})
    return fixes


def build_incident(spec: dict, deploy_id: str | None, *, eval_mode: bool) -> dict:
    inc = {
        # Generation params (version, pool sizes, cert names, ...) are kept on
        # the dict so the log/metric builders can use them. Pydantic ignores
        # extras when the record is validated, so they never reach the JSON.
        **spec,
        "id": spec["id"],
        "family": spec["family"],
        "service": spec["service"],
        "opened_at": spec["opened_at"],
        "alert": build_alert(spec),
        "symptoms": build_symptoms(spec),
        "severity": spec.get("severity", "P1" if spec["family"] in {"expired_tls_cert", "slow_query_migration"} else "P2"),
        "status": "open" if eval_mode else "resolved",
        "root_cause": "" if eval_mode else build_root_cause(spec),
        "fixes_tried": [] if eval_mode else build_fixes(spec),
        "resolution_minutes": 0 if eval_mode else spec.get("resolution_minutes", 30),
        "resolved_by": "" if eval_mode else spec.get("resolved_by", ""),
        "related_deploy_id": deploy_id,
        "log_file": f"data/logs/{spec['id']}.log",
        "metrics_file": f"data/metrics/{spec['id']}.json",
        "tags": [spec["family"], spec["service"]],
        "true_fix": spec.get("true_fix"),
        "novel": bool(spec.get("novel", False)),
    }
    if spec["family"] == "redis_pool_exhaustion":
        # With a deploy, the new (small) pool is live; without one, the old
        # config is still in place. Logs and metrics must agree.
        inc["pool_size"] = spec.get("pool_new") if spec.get("version") else spec.get("pool_old")
    return inc


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    for directory in (LOGS, METRICS):
        for old in directory.glob("*"):
            if old.is_file():
                old.unlink()

    # ---- deploy records (incident-triggered + background noise) ----------
    deploy_events: list[dict] = []
    for spec in HISTORY_SPECS + EVAL_SPECS:
        deploy_events.extend(deploy_plan(spec["id"], spec, utc(spec["opened_at"])))
    for service, version, when, who, summary, commit, previous in NOISE_DEPLOYS:
        deploy_events.append({
            "service": service, "version": version, "deployed_at": when,
            "deployed_by": who, "summary": summary, "commit": commit,
            "previous_version": previous, "incident_id": None,
        })
    deploy_events.sort(key=lambda d: (d["deployed_at"], d["service"]))
    deploys = []
    incident_deploy: dict[str, str] = {}
    for i, item in enumerate(deploy_events, start=1):
        deploy_id = f"DEP-{i:04d}"
        deploys.append({
            "id": deploy_id,
            "service": item["service"],
            "version": item["version"],
            "deployed_at": item["deployed_at"],
            "deployed_by": item["deployed_by"],
            "summary": item["summary"],
            "commit": item["commit"],
            "previous_version": item["previous_version"],
        })
        if item.get("incident_id"):
            incident_deploy[item["incident_id"]] = deploy_id

    # ---- history ---------------------------------------------------------
    history = []
    for spec in HISTORY_SPECS:
        inc = build_incident(spec, incident_deploy.get(spec["id"]), eval_mode=False)
        history.append(inc)
    for inc in history:
        (LOGS / f"{inc['id']}.log").write_text(make_logs(inc), encoding="utf-8")
        (METRICS / f"{inc['id']}.json").write_text(
            json.dumps(make_metrics(inc), indent=2) + "\n", encoding="utf-8")

    # ---- eval ------------------------------------------------------------
    eval_incidents = []
    for spec in EVAL_SPECS:
        inc = build_incident(spec, incident_deploy.get(spec["id"]), eval_mode=True)
        eval_incidents.append(inc)
    for inc in eval_incidents:
        (LOGS / f"{inc['id']}.log").write_text(make_logs(inc), encoding="utf-8")
        (METRICS / f"{inc['id']}.json").write_text(
            json.dumps(make_metrics(inc), indent=2) + "\n", encoding="utf-8")

    # ---- validate with pydantic, then write -------------------------------
    history_models = [Incident.model_validate(i).model_dump(mode="json") for i in history]
    eval_models = [EvalIncident.model_validate(i).model_dump(mode="json") for i in eval_incidents]

    _write(DATA / "services.json", {"services": SERVICES})
    _write(DATA / "deploys.json", {"deploys": deploys})
    _write(DATA / "incidents_history.json", {"incidents": history_models})
    _write(DATA / "incidents_eval.json", {"incidents": eval_models})
    _write(DATA / "families.json", {"families": FAMILIES})

    print(f"services:  {len(SERVICES)}")
    print(f"deploys:   {len(deploys)}")
    print(f"history:   {len(history_models)} incidents")
    print(f"eval:      {len(eval_models)} incidents "
          f"({sum(1 for i in eval_models if i['novel'])} novel)")
    print(f"logs:      {len(list(LOGS.glob('*.log')))} files")
    print(f"metrics:   {len(list(METRICS.glob('*.json')))} snapshots")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
