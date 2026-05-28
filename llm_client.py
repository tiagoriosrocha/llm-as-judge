"""
Cliente para comunicacao com Azure OpenAI.

Fornece chamadas sincrona com retry automatico, parse de JSON e suporte a
schema para uso com DeepEval.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import APIError, APITimeoutError, AzureOpenAI, RateLimitError

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Cliente Azure OpenAI com retry automatico e tratamento de erros."""

    def __init__(self, model_override: str | None = None):
        Config.validate()
        self.client = AzureOpenAI(
            api_key=Config.LLM_API_KEY,
            api_version=Config.LLM_API_VERSION,
            base_url=Config.LLM_ENDPOINT,
        )
        self.model = model_override or Config.LLM_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY_SECONDS
        self.timeout = Config.TIMEOUT_SECONDS
        self.max_tokens = Config.LLM_MAX_TOKENS
        logger.info("LLM Client initialized with model: %s", self.model)

    def _build_request_kwargs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        # Alguns deployments corporativos rejeitam parametros extras em certas rotas.
        # Por isso enviamos apenas os argumentos essenciais por padrao.
        return {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }

    def call_llm(self, messages: list[dict[str, str]]) -> str:
        """Chama o LLM com retry automatico."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("Calling LLM (attempt %s/%s)", attempt, self.max_retries)
                logger.debug("Request messages: %s", messages)

                response = self.client.chat.completions.create(
                    **self._build_request_kwargs(messages)
                )
                content = response.choices[0].message.content or ""
                logger.debug("LLM response received (length: %s)", len(content))
                return content
            except (RateLimitError, APITimeoutError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call failed (attempt %s/%s): %s. Retrying in %ss...",
                        attempt,
                        self.max_retries,
                        type(exc).__name__,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                logger.error("LLM call failed after %s attempts", self.max_retries)
            except APIError as exc:
                logger.error("LLM API error: %s", exc)
                raise

        raise RuntimeError(
            "Failed to call LLM after "
            f"{self.max_retries} attempts. Last error: {last_error}"
        ) from last_error

    def call_llm_for_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Chama o LLM esperando um objeto JSON na resposta."""
        response_text = self.call_llm(messages)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON from response: %s", exc)

        extracted_json = self._extract_json_from_text(response_text)
        if extracted_json is not None:
            logger.debug("Successfully extracted JSON from response text")
            return extracted_json

        logger.error("Invalid JSON response from LLM: %s", response_text[:200])
        raise ValueError(f"LLM returned invalid JSON. Response: {response_text[:500]}")

    def call_llm_for_schema(
        self,
        messages: list[dict[str, str]],
        schema: type[Any],
    ) -> Any:
        """Chama o LLM e converte o JSON retornado para um schema Pydantic."""
        parsed = self.call_llm_for_json(messages)

        if hasattr(schema, "model_validate"):
            return schema.model_validate(parsed)
        if hasattr(schema, "parse_obj"):
            return schema.parse_obj(parsed)

        raise TypeError("Schema fornecido nao parece ser um modelo Pydantic compativel.")

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any] | None:
        """Tenta extrair um dicionario JSON de um texto."""
        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(cleaned[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                return None

        return None
