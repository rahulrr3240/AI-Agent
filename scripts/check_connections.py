"""Verify Hindsight and Groq are actually usable before the demo.

Hindsight: creates a throwaway bank, retains a small postmortem, recalls it,
checks the expected fact came back, then deletes the bank.
Groq:      asks the model for a validated JSON object.

  python scripts/check_connections.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oncall.config import load_settings  # noqa: E402
from oncall.llm import GroqLLM, LLMError  # noqa: E402
from oncall.memory import HindsightMemory, MemoryUnavailable  # noqa: E402

PROBE = (
    "Incident INC-999 (checkout-service, 2026-01-01). Alert: p95 latency > 2000ms. "
    "Symptom: ConnectionPoolTimeoutError in logs after a deploy. "
    "Fixes: restart_service -> FAILED; rollback_deploy_and_increase_pool -> WORKED."
)


class Probe(BaseModel):
    ok: bool
    service: str


def check_hindsight(settings) -> bool:
    print("Hindsight")
    if not settings.hindsight_base_url:
        print("  FAIL  HINDSIGHT_BASE_URL is not set (Cloud: https://api.hindsight.vectorize.io)")
        return False
    bank = f"oncall-conncheck-{uuid.uuid4().hex[:8]}"
    memory = None
    try:
        memory = HindsightMemory(settings, bank_id=bank)
        memory.ensure_bank()
        memory.retain_incident({
            "id": "INC-999",
            "family": "redis_pool_exhaustion",
            "service": "checkout-service",
            "opened_at": "2026-01-01T00:00:00Z",
            "alert": "checkout-service p95 latency > 2000ms for 5m",
            "symptoms": ["p95 latency 4.1s", "ConnectionPoolTimeoutError in logs"],
            "root_cause": "Redis pool exhausted after a deploy reduced pool size.",
            "fixes_tried": [
                {"fix": "restart_service", "outcome": "failed", "note": "temporary"},
                {"fix": "rollback_deploy_and_increase_pool", "outcome": "worked",
                 "note": "rollback + pool 50"},
            ],
            "resolution_minutes": 25,
            "resolved_by": "priya",
            "log_file": "",
            "metrics_file": "",
        })
        hits = memory.recall_similar(
            alert="checkout-service p95 latency > 2000ms",
            service="checkout-service",
            symptoms=["ConnectionPoolTimeoutError"],
            logs_excerpt="ConnectionPoolTimeoutError: Timeout waiting for connection from pool",
            k=3,
        )
        if not hits:
            print("  FAIL  retain succeeded but recall returned nothing")
            return False
        joined = " ".join(hit.text for hit in hits)
        if "INC-999" not in joined and "ConnectionPoolTimeoutError" not in joined:
            print(f"  FAIL  recall returned unrelated memories: {joined[:200]}")
            return False
        print(f"  PASS  retain + recall round trip ({len(hits)} memory group(s))")
        print(f"        {hits[0].text[:160]}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        print("        If this is 401/403, check HINDSIGHT_API_KEY. If it is a timeout, check the URL.")
        return False
    finally:
        if memory is not None:
            try:
                memory.delete_bank()
                print(f"  Cleaned up probe bank {bank}.")
            except Exception:  # noqa: BLE001 - cleanup is best effort
                print(f"  Note: could not delete probe bank {bank}; delete it in the console.")
            memory.close()


def check_groq(settings) -> bool:
    print("Groq")
    llm = GroqLLM(settings)
    if not llm.configured:
        print("  FAIL  GROQ_API_KEY is not set")
        return False
    try:
        result = llm.chat_json(
            [
                {"role": "system", "content": "Answer only with JSON."},
                {"role": "user", "content": (
                    'Return {"ok": true, "service": "checkout-service"} exactly.'
                )},
            ],
            Probe,
            max_tokens=200,
        )
        print(f"  PASS  {result.model} returned valid JSON in {result.latency_ms}ms "
              f"({result.attempts} attempt(s), {result.prompt_tokens}+{result.completion_tokens} tokens)")
        print(f"        {result.data.model_dump()}")
        return True
    except LLMError as exc:
        print(f"  FAIL  {exc}")
        return False


def main() -> int:
    settings = load_settings()
    print(f"Models: {settings.groq_model} (fallback {settings.groq_fallback_model})")
    print(f"Bank:   {settings.hindsight_bank_id}")
    print()
    hindsight_ok = check_hindsight(settings)
    print()
    groq_ok = check_groq(settings)
    print()
    if hindsight_ok and groq_ok:
        print("All checks passed. You can run: streamlit run app.py")
        return 0
    print("One or more checks failed. Fix the items above; the app still runs with clearly-labelled "
          "fallbacks, but the benchmark will not reflect real Hindsight/Groq behavior.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
