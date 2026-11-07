from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from oncall.config import Settings
from oncall.llm import GroqLLM, LLMError, extract_json, strip_reasoning


class Probe(BaseModel):
    ok: bool
    service: str


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        content = item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def settings() -> Settings:
    return Settings(groq_api_key="test-key", groq_model="primary-model",
                    groq_fallback_model="fallback-model")


def llm_with(responses) -> tuple[GroqLLM, FakeClient]:
    llm = GroqLLM(settings())
    fake = FakeClient(responses)
    llm._client = fake
    return llm, fake


MESSAGES = [{"role": "user", "content": "answer"}]
VALID = '{"ok": true, "service": "checkout-service"}'


def test_repair_retry_after_invalid_json():
    llm, fake = llm_with(["this is not json", VALID])
    result = llm.chat_json(MESSAGES, Probe)
    assert result.data.ok is True
    assert result.attempts == 2
    assert result.repaired is True
    assert result.model == "primary-model"
    # the repair message must include the validation problem and the schema
    repair = fake.completions.calls[1]["messages"][-1]["content"]
    assert "invalid" in repair.lower() and "schema" in repair.lower()


def test_repair_retry_after_schema_violation():
    llm, _ = llm_with(['{"ok": "yes"}', VALID])
    result = llm.chat_json(MESSAGES, Probe)
    assert result.attempts == 2
    assert result.repaired


def test_falls_back_to_second_model_when_primary_keeps_failing():
    llm, fake = llm_with(["nope", "still nope", VALID])
    result = llm.chat_json(MESSAGES, Probe)
    assert result.model == "fallback-model"
    assert result.attempts == 3
    assert any("fallback" in note for note in result.notes)
    assert [call["model"] for call in fake.completions.calls] == [
        "primary-model", "primary-model", "fallback-model"]


def test_raises_clear_error_when_all_attempts_fail():
    llm, _ = llm_with(["bad", "bad", "bad"])
    with pytest.raises(LLMError) as excinfo:
        llm.chat_json(MESSAGES, Probe)
    assert "no valid Probe" in str(excinfo.value)


def test_rate_limit_backoff_then_success(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("oncall.llm.time.sleep", lambda s: sleeps.append(s))
    llm, _ = llm_with([Exception("429 rate limit exceeded"), VALID])
    result = llm.chat_json(MESSAGES, Probe)
    assert result.data.service == "checkout-service"
    assert sleeps and sleeps[0] >= 2.0


def test_missing_api_key_raises_immediately():
    llm = GroqLLM(Settings(groq_api_key=""))
    with pytest.raises(LLMError):
        llm.chat_json(MESSAGES, Probe)


def test_reasoning_tags_and_fences_are_stripped():
    raw = '<think>the user wants JSON</think>\n```json\n{"ok": true, "service": "auth"}\n```'
    assert extract_json(raw) == '{"ok": true, "service": "auth"}'
    assert "think" not in strip_reasoning(raw)


def test_json_mode_is_requested_but_not_required():
    llm, fake = llm_with([VALID])
    llm.chat_json(MESSAGES, Probe)
    assert fake.completions.calls[0]["response_format"] == {"type": "json_object"}
