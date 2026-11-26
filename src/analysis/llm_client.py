"""Единственный модуль, который общается с Gemini API.

Остальной код зависит только от протокола ``LLMClient`` — поэтому в тестах
клиент подменяется заглушкой, а реальный ключ не нужен.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Вызов LLM не удался (после допустимого числа повторов)."""


class LLMOutputError(LLMError):
    """LLM вернула ответ, не прошедший валидацию схемы."""


class LLMClient(Protocol):
    model: str
    calls: int

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T: ...


def extract_json(raw: str) -> str:
    """Достаёт JSON-объект из ответа (убирает ```json-ограждения и лишний текст)."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("В ответе нет JSON-объекта")
    return text[start:end + 1]


def parse_model(raw: str | None, schema: type[T]) -> T:
    """Строгая валидация ответа LLM Pydantic-схемой. Бросает ValueError/ValidationError."""
    if not raw or not raw.strip():
        raise ValueError("Пустой ответ модели")
    return schema.model_validate_json(extract_json(raw))


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        requests_per_minute: int = 10,
        max_api_retries: int = 4,
        max_json_retries: int = 2,
        backoff_base: float = 4.0,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.calls = 0
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._max_api_retries = max_api_retries
        self._max_json_retries = max_json_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._clock = clock
        self._last_call = float("-inf")
        if sdk_client is not None:
            self._client = sdk_client
        else:
            from google import genai  # импорт здесь: тесты без SDK-вызовов не требуют сети
            self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------ #
    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        feedback = ""
        last_error: Exception | None = None
        for attempt in range(self._max_json_retries + 1):
            raw = self._call(system, prompt + feedback, schema)
            try:
                return parse_model(raw, schema)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                log.warning("Невалидный JSON от модели (попытка %d/%d): %s",
                            attempt + 1, self._max_json_retries + 1, _short(exc))
                feedback = (
                    "\n\nВНИМАНИЕ: предыдущий ответ не прошёл проверку схемы: "
                    f"{_short(exc)}\nВерни ТОЛЬКО корректный JSON по схеме."
                )
        raise LLMOutputError(f"Ответ модели не прошёл валидацию после {self._max_json_retries + 1} попыток: "
                             f"{_short(last_error)}")

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        wait = self._min_interval - (self._clock() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = self._clock()

    def _call(self, system: str, prompt: str, schema: type[BaseModel]) -> str | None:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        )
        for attempt in range(self._max_api_retries + 1):
            self._throttle()
            self.calls += 1
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                return resp.text
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code not in RETRYABLE_HTTP or attempt == self._max_api_retries:
                    raise LLMError(f"Ошибка Gemini API (код {code}): {_short(exc)}") from exc
                delay = self._backoff_base * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Gemini API %s — повтор через %.1f с (попытка %d/%d)",
                            code, delay, attempt + 1, self._max_api_retries)
                self._sleep(delay)
            except (ConnectionError, TimeoutError, OSError) as exc:
                if attempt == self._max_api_retries:
                    raise LLMError(f"Сетевая ошибка при обращении к Gemini: {_short(exc)}") from exc
                delay = self._backoff_base * (2 ** attempt)
                log.warning("Сетевая ошибка (%s) — повтор через %.1f с", _short(exc), delay)
                self._sleep(delay)
        raise LLMError("Недостижимое состояние повторов")  # pragma: no cover


def _short(exc: object, limit: int = 300) -> str:
    s = str(exc).replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def dump_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
