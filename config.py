"""
Configuracao centralizada do projeto.

Este modulo carrega e valida as variaveis de ambiente usadas pela avaliacao
com Azure OpenAI no ambiente de producao.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Carrega .env.local antes de .env quando existirem.
load_dotenv(PROJECT_ROOT / ".env.local", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Config:
    """Configuracoes da aplicacao."""

    BASE_DIR: Path = PROJECT_ROOT
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # ====== CONFIGURACOES AZURE OPENAI ======
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_ENDPOINT: str = os.getenv("LLM_ENDPOINT", "")
    LLM_API_VERSION: str = os.getenv("LLM_API_VERSION", "2025-01-01-preview")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-4-petrobras")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "400000"))

    # ====== CONFIGURACOES DE PROCESSAMENTO ======
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", "2"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "0"))

    # ====== METADADOS / LOGGING ======
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "1.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def validate(cls) -> None:
        """Valida configuracoes obrigatorias e prepara diretorios."""
        if not cls.LLM_API_KEY:
            raise ValueError("LLM_API_KEY nao foi configurada no arquivo .env")
        if not cls.LLM_ENDPOINT:
            raise ValueError("LLM_ENDPOINT nao foi configurada no arquivo .env")
        if not cls.LLM_MODEL:
            raise ValueError("LLM_MODEL nao foi configurada no arquivo .env")
        if not 0.0 <= cls.TEMPERATURE <= 2.0:
            raise ValueError(
                f"TEMPERATURE deve estar entre 0.0 e 2.0, recebido: {cls.TEMPERATURE}"
            )
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls) -> dict[str, object]:
        """Retorna configuracoes nao sensiveis."""
        return {
            "LLM_ENDPOINT": cls.LLM_ENDPOINT,
            "LLM_API_VERSION": cls.LLM_API_VERSION,
            "LLM_MODEL": cls.LLM_MODEL,
            "LLM_MAX_TOKENS": cls.LLM_MAX_TOKENS,
            "TEMPERATURE": cls.TEMPERATURE,
            "MAX_RETRIES": cls.MAX_RETRIES,
            "RETRY_DELAY_SECONDS": cls.RETRY_DELAY_SECONDS,
            "TIMEOUT_SECONDS": cls.TIMEOUT_SECONDS,
            "MAX_QUESTIONS": cls.MAX_QUESTIONS,
            "PROMPT_VERSION": cls.PROMPT_VERSION,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "OUTPUT_DIR": str(cls.OUTPUT_DIR),
            "LOGS_DIR": str(cls.LOGS_DIR),
        }
