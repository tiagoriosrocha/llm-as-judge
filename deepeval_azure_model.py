"""
Wrapper do DeepEval para usar o cliente AzureOpenAI customizado do projeto.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM

from llm_client import LLMClient


class AzureDeepEvalModel(DeepEvalBaseLLM):
    """Modelo customizado para uso do DeepEval com o cliente Azure da aplicacao."""

    def __init__(self, model_override: str | None = None):
        self._client = LLMClient(model_override=model_override)

    def load_model(self) -> LLMClient:
        return self._client

    def generate(self, prompt: str, schema: type[Any] | None = None) -> Any:
        messages = [{"role": "user", "content": prompt}]
        client = self.load_model()

        if schema is not None:
            return client.call_llm_for_schema(messages, schema)
        return client.call_llm(messages)

    async def a_generate(self, prompt: str, schema: type[Any] | None = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.load_model().model
