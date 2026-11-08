"""Groq client for structured, validated JSON reasoning.

Reliability ladder for every call:
  1. primary model
  2. primary model + a repair instruction containing the validation error
  3. fallback model + the same repair instruction
  4. raise LLMError (the agent then uses its deterministic analyzer)

With ONCALL_DEBUG_LLM=1, every attempt that fails to parse or validate is written
to .llm_failures/ (model, error, raw output) for inspection. Default runs write
nothing, and neither the prompt nor the API key is ever recorded.

Rate limits (429) are retried with exponential backoff. Reasoning tags some
models emit (`<think>...</think>`, ```` ```json ```` fences) are stripped before
parsing. Token usage and latency are returned so the UI can show them.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from oncall.config import BASE_DIR, Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """No usable model response after every retry."""


DEBUG_ENV = "ONCALL_DEBUG_LLM"
FAILURE_DIR = Path(BASE_DIR) / ".llm_failures"


def describe_validation_error(exc: ValidationError) -> str:
    """Name the fields that failed, so the repair prompt and the final error are specific."""
    problems = []
    for error in exc.errors()[:6]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        problems.append(f"{location}: {error.get('msg', 'invalid')} [{error.get('type', '')}]")
    return "schema validation failed: " + ("; ".join(problems) or "no detail")


def capture_failure(model: str, error: str, raw: str, schema_name: str,
                    attempt: int, finish_reason: str = "",
                    completion_tokens: int = 0) -> Path | None:
    """Write one failed attempt to .llm_failures/ when ONCALL_DEBUG_LLM=1.

    Only what the model produced is recorded: the request messages (prompt) and
    the API key are never written. Diagnostics must never break a call, so an I/O
    problem is reported and ignored.
    """
    if os.environ.get(DEBUG_ENV) != "1":
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = FAILURE_DIR / f"{stamp}-attempt{attempt}-{schema_name}.txt"
    try:
        FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"model: {model}\n"
            f"schema: {schema_name}\n"
            f"attempt: {attempt}\n"
            f"error: {error}\n"
            f"finish_reason: {finish_reason or 'unknown'}\n"
            f"completion_tokens: {completion_tokens}\n"
            f"raw_chars: {len(raw)}\n"
            "--- raw model output ---\n"
            f"{raw}\n"
            "--- end raw model output ---\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[llm] could not write {path}: {exc}", file=sys.stderr)
        return None
    print(f"[llm] captured failed attempt -> {path}", file=sys.stderr)
    return path


@dataclass
class ToolMessage:
    content: str
    tool_calls: list[dict[str, Any]]
    model: str


@dataclass
class LLMResult:
    data: BaseModel
    model: str
    latency_ms: int
    attempts: int
    repaired: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""
    notes: list[str] = field(default_factory=list)


def strip_reasoning(text: str) -> str:
    """Remove think tags and markdown fences a reasoning model may emit."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text.strip())
    return text.strip()


def extract_json(text: str) -> str:
    cleaned = strip_reasoning(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return cleaned[start:end + 1]


def schema_hint(schema: type[BaseModel]) -> str:
    """A compact field/type/enum skeleton.

    The full `model_json_schema()` is ~5k characters of `$defs` and JSON-Schema
    keywords; on a free-tier token budget that is most of the request. This
    renders the same contract in a few hundred characters.
    """
    js = schema.model_json_schema()
    defs = js.get("$defs", {})

    def render(node: dict[str, Any], depth: int = 0) -> str:
        node = dict(node)
        seen = 0
        while "$ref" in node and seen < 5:
            node = dict(defs.get(node["$ref"].split("/")[-1], {}))
            seen += 1
        if "anyOf" in node:
            parts = [render(x, depth) for x in node["anyOf"]]
            return "|".join(p for p in parts if p)
        if node.get("type") == "array":
            return f"[{render(node.get('items', {}), depth + 1)}]"
        if node.get("type") == "object" or "properties" in node:
            props = ", ".join(
                f"{key}: {render(value, depth + 1)}"
                for key, value in node.get("properties", {}).items())
            return "{" + props + "}"
        if "enum" in node:
            return "|".join(f'"{value}"' for value in node["enum"])
        return str(node.get("type", "any"))

    return render(js)


class GroqLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    @property
    def configured(self) -> bool:
        return self.settings.has_groq

    def _get_client(self):
        if self._client is None:
            if not self.configured:
                raise LLMError("GROQ_API_KEY is not set")
            from groq import Groq

            self._client = Groq(api_key=self.settings.groq_api_key, timeout=60.0, max_retries=0)
        return self._client

    # -- one model attempt --------------------------------------------------
    def _complete(self, model: str, messages: list[dict[str, Any]],
                  max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(3):  # 429 backoff inside a single model attempt
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                try:
                    response = client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs
                    )
                except Exception as exc:  # noqa: BLE001 - model may reject JSON mode
                    if "response_format" not in str(exc).lower() and "json" not in str(exc).lower():
                        raise
                    response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                content = choice.message.content or ""
                usage = getattr(response, "usage", None)
                return content, {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    # "length" means the object was cut off mid-answer, which is a
                    # different failure from the model choosing to stop early.
                    "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc).lower()
                if "429" in message or "rate limit" in message:
                    time.sleep(2.0 * (2 ** attempt))
                    continue
                raise
        raise LLMError(f"{model} kept hitting rate limits: {last_error}")

    def chat_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   model: str | None = None) -> ToolMessage:
        """One function-calling turn (used only by agentic mode)."""
        client = self._get_client()
        chosen = model or self.settings.groq_model
        try:
            response = client.chat.completions.create(
                model=chosen,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=900,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"tool-calling request failed: {exc}") from exc
        message = response.choices[0].message
        calls: list[dict[str, Any]] = []
        for call in (getattr(message, "tool_calls", None) or []):
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise LLMError(f"malformed tool arguments from {chosen}: {call.function.arguments}") from exc
            calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
        return ToolMessage(content=message.content or "", tool_calls=calls, model=chosen)

    # -- public API ---------------------------------------------------------
    def chat_json(self, messages: list[dict[str, Any]], schema: type[T], *,
                  max_tokens: int = 1800, temperature: float = 0.0,
                  model: str | None = None) -> LLMResult:
        """Call Groq, parse JSON, validate against `schema`, retrying as needed."""
        started = time.perf_counter()
        if not self.configured:
            raise LLMError("GROQ_API_KEY is not set")

        primary = model or self.settings.groq_model
        fallback = self.settings.groq_fallback_model
        plan = [primary, primary, fallback]
        repair_instructions = [False, True, True]

        attempts = 0
        repaired = False
        last_error = "no attempt made"
        last_raw = ""
        prompt_tokens = completion_tokens = 0

        for attempt_model, needs_repair in zip(plan, repair_instructions):
            attempt_messages = list(messages)
            if needs_repair and last_raw:
                attempt_messages = messages + [
                    {"role": "assistant", "content": last_raw[:2000]},
                    {"role": "user", "content": (
                        "Your previous response was invalid: " + str(last_error)[:600] + "\n"
                        "Return ONLY one valid JSON object that matches this JSON Schema. "
                        "No prose, no markdown fences, no reasoning tags.\n"
                        + schema_hint(schema)
                    )},
                ]
            attempts += 1
            try:
                raw, usage = self._complete(attempt_model, attempt_messages, max_tokens, temperature)
            except LLMError as exc:
                last_error = str(exc)
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            last_raw = raw
            finish_reason = str(usage.get("finish_reason", "") or "")
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            try:
                payload = json.loads(extract_json(raw))
            except ValueError as exc:  # no JSON object at all, or malformed JSON
                last_error = f"invalid JSON: {exc}"
                capture_failure(attempt_model, last_error, raw, schema.__name__, attempts,
                                finish_reason, usage["completion_tokens"])
                continue
            try:
                validated = schema.model_validate(payload)
            except ValidationError as exc:
                # ValidationError subclasses ValueError, so it must be caught on
                # its own after the JSON step: one combined `except ValueError`
                # reports every missing field as "invalid JSON" and the repair
                # prompt then cannot name the field.
                last_error = describe_validation_error(exc)
                capture_failure(attempt_model, last_error, raw, schema.__name__, attempts,
                                finish_reason, usage["completion_tokens"])
                continue

            repaired = needs_repair
            latency_ms = int((time.perf_counter() - started) * 1000)
            notes = []
            if finish_reason == "length":
                notes.append("model stopped at the token limit (finish_reason=length)")
            if attempt_model != primary:
                notes.append(f"primary model failed; answered by fallback {attempt_model}")
            if repaired:
                notes.append("answer required a repair retry")
            return LLMResult(
                data=validated,
                model=attempt_model,
                latency_ms=latency_ms,
                attempts=attempts,
                repaired=repaired,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw=last_raw,
                notes=notes,
            )

        raise LLMError(
            f"Groq returned no valid {schema.__name__} after {attempts} attempts. Last error: {last_error}"
        )
