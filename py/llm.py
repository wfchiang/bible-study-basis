from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from config import httpx_client, httpx_async_client, web_search_tool
from data_tools import get_bible_verses, search_article_chunks, search_bible_chunks


async def create_llm_with_tools(
        model: str = "gpt-4.1-mini-2025-04-14",
        use_data_tools: bool = True, use_web_search: bool = False) -> tuple[BaseChatModel, list[BaseTool]]:
    tools: list[BaseTool] = []
    if use_data_tools:
        tools.append(get_bible_verses)
        tools.append(search_article_chunks)
        tools.append(search_bible_chunks)

    if use_web_search and web_search_tool:
        tools.append(web_search_tool)

    llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14",
                    http_client=httpx_client,
                    http_async_client=httpx_async_client,)
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools