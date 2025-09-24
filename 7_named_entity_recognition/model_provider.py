from __future__ import annotations

from openai import AsyncOpenAI
from agents import ModelProvider, Model, OpenAIChatCompletionsModel

from config import settings


def get_gemini_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)


class GeminiModelProvider(ModelProvider):
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.model_name
        self._client = get_gemini_client()

    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or self._model_name,
            openai_client=self._client,
        )


GEMINI_PROVIDER = GeminiModelProvider()


