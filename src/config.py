"""Конфигурация агента: переменные окружения и версия."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

AGENT_VERSION = "0.2.0"
# Единственное место, где задана модель по умолчанию.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RPM = 15


class ConfigError(RuntimeError):
    """Ошибка конфигурации (например, нет API-ключа)."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    requests_per_minute: int


def get_settings(model_override: str | None = None, rpm_override: int | None = None) -> Settings:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise ConfigError(
            "Не задан GEMINI_API_KEY. Получите ключ в Google AI Studio "
            "(https://aistudio.google.com/apikey) и добавьте его в файл .env "
            "(см. .env.example)."
        )
    model = model_override or os.getenv("GEMINI_MODEL", "").strip()
    if not model or model == "your_supported_model_name":
        model = DEFAULT_MODEL
    try:
        rpm = rpm_override or int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", DEFAULT_RPM))
    except ValueError as exc:
        raise ConfigError("GEMINI_REQUESTS_PER_MINUTE должен быть целым числом") from exc
    return Settings(api_key=api_key, model=model, requests_per_minute=max(1, rpm))


def get_agent_version() -> str:
    """Версия агента; добавляет короткий хеш коммита, если проект в git-репозитории."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=3,
        )
        commit = out.stdout.strip()
        if out.returncode == 0 and commit:
            worktree = subprocess.run(
                ["git", "--no-optional-locks", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=3,
            )
            suffix = ", локальные изменения" if worktree.returncode == 0 and worktree.stdout.strip() else ""
            return f"{AGENT_VERSION} (commit {commit}{suffix})"
    except (OSError, subprocess.SubprocessError):
        pass
    return AGENT_VERSION
