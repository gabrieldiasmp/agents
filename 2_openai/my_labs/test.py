from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
from tavily import TavilyClient

from openai import AsyncOpenAI  # ✅ We will use Gemini through the OpenAI-compatible endpoint
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)


load_dotenv(override=True)

# ==========================
# Environment Variables
# ==========================
# Gemini’s OpenAI-compatible endpoint
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("GOOGLE_API_KEY") or ""   # <-- Your Gemini API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = "gemini-2.0-flash"               # <-- Gemini model

if not API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your environment.")

# ==========================
# Create a Gemini Client
# ==========================
# This tells the SDK to use Gemini instead of OpenAI
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Optional: disable OpenAI tracing if you don’t have an OpenAI key
set_tracing_disabled(disabled=True)

# ==========================
# Custom Model Provider
# ==========================
class GeminiModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or MODEL_NAME,
            openai_client=client   # <-- use our Gemini client here
        )

GEMINI_PROVIDER = GeminiModelProvider()

# ==========================
# Tools
# ==========================

# --- 1. Tavily tool ---
tavily = TavilyClient(api_key=TAVILY_API_KEY)

@function_tool
def tavily_search(query: str) -> str:
    """
    Perform a web search using Tavily and return the best 3 results.
    """
    result = tavily.search(query=query, max_results=3)
    return result

# ==========================
# Run an Agent
# ==========================
async def main():

    # --- 3. Build the agent ---
    agent = Agent(
        name="WebSearchAgent",
        instructions="You are a helpful assistant. Use the Tavily tool to answer web queries.",
        tools=[tavily_search],
    )

    # Use the Gemini provider explicitly
    result = await Runner.run(
        agent,
        "Find the latest news about generative AI funding rounds",
        run_config=RunConfig(model_provider=GEMINI_PROVIDER),
    )

    print("\n--- FINAL OUTPUT ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
