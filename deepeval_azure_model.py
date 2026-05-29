"""
Wrapper do DeepEval para usar o cliente AzureOpenAI customizado do projeto.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM

from llm_client import LLMClient

DEEPEVAL_SYSTEM_MESSAGE = (
    "You are an evaluation assistant. Follow the user's requested JSON schema exactly. "
    "Whenever the output contains a field named 'reason' or any explanatory justification, "
    "write that explanation in English. Preserve Portuguese source text, technical terms, "
    "and direct quotes when they are needed as evidence."
)


class AzureDeepEvalModel(DeepEvalBaseLLM):
    """Modelo customizado para uso do DeepEval com o cliente Azure da aplicacao."""

    def __init__(self, model_override: str | None = None):
        self._client = LLMClient(model_override=model_override)

    def load_model(self) -> LLMClient:
        return self._client

    def generate(self, prompt: str, schema: type[Any] | None = None) -> Any:
        messages = [
            {"role": "system", "content": DEEPEVAL_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        client = self.load_model()

        if schema is not None:
            return client.call_llm_for_schema(messages, schema)
        return client.call_llm(messages)

    async def a_generate(self, prompt: str, schema: type[Any] | None = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.load_model().model
