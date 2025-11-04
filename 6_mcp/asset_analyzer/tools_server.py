import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

mcp = FastMCP("asset_tools_server")


class TavilySearchArgs(BaseModel):
    query: str = Field(description="Search query to pass to Tavily")


@mcp.tool()
def tavily_search(args: TavilySearchArgs) -> str:
    """Search the web using Tavily API and return a concise JSON/text response."""
    if not TAVILY_API_KEY:
        return "Tavily API key is not configured."
    url = "https://api.tavily.com/search"
    try:
        resp = requests.post(
            url,
            json={"api_key": TAVILY_API_KEY, "query": args.query, "max_results": 5},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"tavily_search error: {e}"


class PolygonPriceArgs(BaseModel):
    symbol: str = Field(description="Ticker symbol, e.g., AAPL")


@mcp.tool()
def polygon_price(args: PolygonPriceArgs) -> str:
    """Get the most recent daily close price for the given symbol via Polygon."""
    if not POLYGON_API_KEY:
        return "Polygon API key is not configured."
    url = f"https://api.polygon.io/v2/aggs/ticker/{args.symbol}/prev"
    try:
        resp = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"polygon_price error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")


