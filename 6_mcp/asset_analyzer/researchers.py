import os
import json
from contextlib import AsyncExitStack
from typing import TypedDict, Optional, List

from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from .mcp_params import researcher_mcp_server_params
from .templates import researcher_instructions
from .strategies import default_strategies

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)


class ResearchState(TypedDict):
    topic: str
    query: str
    research: Optional[str]
    suggestion: Optional[str]


class AssetResearcher:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.worker_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self._graph = None
        self._search_server = None

    def _build_graph(self):
        if self._graph is not None:
            return self._graph

        async def prepare_query(state: ResearchState) -> ResearchState:
            if state.get("query"):
                return state
            topic = state.get("topic") or "interesting investment opportunities"
            q = f"Latest investable assets related to: {topic}. Identify symbols or indices and key theses."
            return {**state, "query": q}

        async def search(state: ResearchState) -> ResearchState:
            text = None

            if self._search_server:
                try:
                    tools = await self._search_server.list_tools()

                    # find a search-like tool
                    search_tool_name = None
                    for t in tools:
                        n = getattr(t, "name", "").lower()
                        if "search" in n:
                            search_tool_name = t.name
                            break

                    # call the search tool
                    if search_tool_name:
                        result = await self._search_server.call_tool(search_tool_name, {"query": state["query"]})

                        # FIX: make result JSON-serializable
                        # (convert CallToolResult or any complex object to a safe structure)
                        if hasattr(result, "__dict__"):
                            text = result.__dict__
                        elif isinstance(result, (list, dict, str, int, float, bool)) or result is None:
                            text = result
                        else:
                            text = str(result)

                except Exception as e:
                    text = f"Search error: {e}"

            # FIX: ensure we can serialize cleanly
            try:
                research_str = json.dumps(text)
            except TypeError:
                research_str = json.dumps(str(text))

            return {**state, "research": research_str if text is not None else None}

        async def reason(state: ResearchState) -> ResearchState:
            # Combine global researcher instructions with persona
            system = researcher_instructions() + "\n\nPersona focus: " + self.instructions
            user = (
                "Using the following research context (JSON or text), propose a single suggestion in JSON with keys: "
                "asset (string), decision (string), reason (string).\n\n"
                f"Context: {state.get('research') or ''}"
            )
            resp = await self.worker_llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            content = resp.content if hasattr(resp, "content") else str(resp)
            # Try to reduce to JSON only
            try:
                txt = content.strip()
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    br = txt.find("{")
                    if br >= 0:
                        txt = txt[br:]
                json.loads(txt)  # validate
                content = txt
            except Exception:
                pass
            return {**state, "suggestion": content}

        graph = StateGraph(ResearchState)
        graph.add_node("prepare_query", prepare_query)
        graph.add_node("search", search)
        graph.add_node("reason", reason)
        graph.add_edge(START, "prepare_query")
        graph.add_edge("prepare_query", "search")
        graph.add_edge("search", "reason")
        graph.add_edge("reason", END)
        self._graph = graph.compile()
        return self._graph

    async def run(self, topic: str = "markets") -> str:
        graph = self._build_graph()
        async with AsyncExitStack() as stack:
            researcher_servers = [
                await stack.enter_async_context(
                    MCPServerStdio(params, client_session_timeout_seconds=120)
                )
                for params in researcher_mcp_server_params(self.name)
            ]
            # Also add local tools server (Tavily + Polygon) for convenience
            try:
                tools_srv = await stack.enter_async_context(
                    MCPServerStdio({"command": "uv", "args": ["run", "asset_analyzer/tools_server.py"]}, client_session_timeout_seconds=120)
                )
                researcher_servers.append(tools_srv)
            except Exception:
                pass
            # Find a search-capable server
            for srv in researcher_servers:
                try:
                    tools = await srv.list_tools()
                    if any("search" in getattr(t, "name", "").lower() for t in tools):
                        self._search_server = srv
                        break
                except Exception:
                    pass
            state: ResearchState = {"topic": topic, "query": "", "research": None, "suggestion": None}
            result = await graph.ainvoke(state)
            return result.get("suggestion") or "{}"


def default_researchers() -> List[AssetResearcher]:
    researchers: List[AssetResearcher] = []
    for name, instr in default_strategies():
        researchers.append(AssetResearcher(name=name, instructions=instr))
    return researchers


