import os
import logging
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import web_search_tool
# from data_tools import get_bible_verses, search_article_chunks, search_bible_chunks


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

skills_folder = str(Path(__file__).parent / "skills")

def check_env_vars() -> None:
    """
    Checks if the required environment variables are set.
    Raises an exception if any are missing.
    """
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")


async def create_agent():
    check_env_vars()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0,
    )

    agent = create_deep_agent(
        model=llm,
        # tools=[
        #     get_bible_verses,
        #     search_article_chunks,
        #     search_bible_chunks
        # ]
        tools=[web_search_tool] if web_search_tool else [],
        skills=[skills_folder],
    )

    return agent


if __name__ == "__main__":
    import asyncio
    async def main():
        agent = await create_agent()

        inputs = {
            "messages": [
                {"role": "user", "content": "遵守神的命令會帶來真正的喜樂嗎?為什麼?"}
            ]
        }

        for chunk in agent.stream(inputs, stream_mode="updates", subgraphs=True, version="v2"):
            if chunk["type"] == "updates":
                if chunk["ns"]:
                    print(f"\n[Sub-Agent Step Update in namespace {chunk['ns']}]:")
                    print(chunk["data"])
                else:
                    print("\n[Main Agent Step Update]:")
                    print(chunk["data"])

    asyncio.run(main())
