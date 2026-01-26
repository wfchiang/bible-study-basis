import os
from pathlib import Path

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain_tavily import TavilySearch
import yaml


# Load configuration
config_file_path = os.environ.get(
    "BSB_CONFIG_PATH", str(Path(__file__).parents[1] / "config.yaml"))
config = None
with open(config_file_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# Load the embedding model
embedding_model = OpenAIEmbeddings(
        model=config["embedding"]["openai_model"],
        chunk_size=config["embedding"]["openai_batch_size"],
        max_retries=config["embedding"]["openai_max_retries"])

embedding_length = len(embedding_model.embed_query("This is a test"))


# Create the MCP client
if "BSB_MCP_SERVER" in os.environ:
    mcp_client = MultiServerMCPClient({
        "service": {
            "transport": config["mcp"]["transport"],
            "url": os.environ["BSB_MCP_SERVER"],
        }
    })
else:
    mcp_client = None


# Create the HTTP clients
class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # To fix an OpenAI issue: Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)

class CustomHTTPAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # To fix an OpenAI issue: Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)

httpx_client = CustomHTTPClient()
httpx_async_client = CustomHTTPAsyncClient()


# Create a web search tool
def create_web_search_tool() -> BaseTool | None:
    """
    Seek for the environment variables and return the web search tool.
    """
    if "TAVILY_API_KEY" in os.environ:
        return TavilySearch(max_results=3)
    return None

web_search_tool = create_web_search_tool()
