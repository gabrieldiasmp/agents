import os
from dotenv import load_dotenv


load_dotenv(override=True)

tvly_api_key = os.getenv("TAVILY_API_KEY")


def researcher_mcp_server_params(name: str):
    return [
        {
            "command": "mcp-remote",
            "args": [
                f"https://mcp.tavily.com/mcp/?tavilyApiKey={tvly_api_key}",
            ],
        },
    ]


