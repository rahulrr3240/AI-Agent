"""Configuration loaded from environment / .env.

No secret ever lives in the repository: only .env.example is committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_FALLBACK_MODEL = "qwen/qwen3-32b"
HINDSIGHT_CLOUD_URL = "https://api.hindsight.vectorize.io"


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    groq_model: str = DEFAULT_MODEL
    groq_fallback_model: str = DEFAULT_FALLBACK_MODEL
    hindsight_base_url: str = ""
    hindsight_api_key: str = ""
    hindsight_bank_id: str = "oncall-demo"
    hindsight_timeout: float = 60.0
    agent_mode: str = "deterministic"
    memory_backend: str = "auto"  # auto | hindsight | local
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR

    # --- capability checks -------------------------------------------------
    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def has_hindsight(self) -> bool:
        return bool(self.hindsight_base_url.strip())

    @property
    def groq_models(self) -> list[str]:
        """Primary model first, then the fallback (deduplicated)."""
        models = [self.groq_model.strip(), self.groq_fallback_model.strip()]
        return [m for i, m in enumerate(models) if m and m not in models[:i]]


def load_settings(env_file: Path | None = None) -> Settings:
    env_path = env_file or (BASE_DIR / ".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)

    def _get(name: str, default: str = "") -> str:
        return (os.getenv(name) or default).strip()

    try:
        timeout = float(_get("HINDSIGHT_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0

    return Settings(
        groq_api_key=_get("GROQ_API_KEY"),
        groq_model=_get("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        groq_fallback_model=_get("GROQ_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        or DEFAULT_FALLBACK_MODEL,
        hindsight_base_url=_get("HINDSIGHT_BASE_URL"),
        hindsight_api_key=_get("HINDSIGHT_API_KEY"),
        hindsight_bank_id=_get("HINDSIGHT_BANK_ID", "oncall-demo") or "oncall-demo",
        hindsight_timeout=timeout,
        agent_mode=_get("AGENT_MODE", "deterministic").lower() or "deterministic",
        memory_backend=_get("MEMORY_BACKEND", "auto").lower() or "auto",
        base_dir=BASE_DIR,
        data_dir=DATA_DIR,
        reports_dir=REPORTS_DIR,
    )
