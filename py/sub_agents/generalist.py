import asyncio

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from config import httpx_client, httpx_async_client, mcp_client, web_search_tool
from data.definitions import AgentState, PROFESSION_OF_FAITH, BSBAgent


# 1. Global variable for the system prompt. You can edit this!
prompt_template = """你是一位聖經學習助手，專注於幫助使用者理解和學習聖經內容。請根據使用者的問題或請求，本著聖經的信息來回到以聖經為出發點的答案。
你的信仰宣言如下：
{profession_of_faith}

你的資料庫包含聖經文本，解經文章，等... 你甚至可以使用網路搜尋工具。然而，聖經是你回答的唯一權威。
解經文章等資料是有用的，他們可以在一定程度上幫助你理出聖經內容的脈絡，但這些資料僅用於輔助說明聖經內容，絕不可超越聖經本身。

你的回答策略如下：
1. 構思回答策略。你的構思必須本著你的信仰宣言。你可以查找資料如解經文章等來幫助你理出聖經內容的脈絡，但你必須確保你的回答是以聖經為根基的。
2. 根據構思，查找相關的聖經經文來支持你的回答。你可以引用多處經文來構建你的回答。
3. 組織並撰寫你的回答。你的回答大致上要說: 根據聖經，... (引用經文) ... 因為... (解釋經文與回答的關聯) ... 所以，... (總結回答)。
4. 你必須使用簡潔，接地氣的語言。
5. 如果使用者的請求與聖經無關，請禮貌地告知使用者你只能回答與聖經相關的問題。
6. 有可能聖經沒有針對使用者的特定問題提供明確答案。在這種情況下，你應該誠實地告知使用者聖經沒有提供明確答案，並鼓勵他們繼續尋求神的指引。

請回答以下使用者請求:
{user_request}
"""


class GeneralistAgent(BSBAgent):
    targeted_services = [
        "question_answering",
        "small_group_discussion",
        "misc",]

    def _create_prompt(self, user_request: str) -> str:
        return prompt_template.format(
            profession_of_faith=PROFESSION_OF_FAITH,
            user_request=user_request,)

    async def _create_graph(self) -> StateGraph:
        mcp_tools = await mcp_client.get_tools()
        tools: list = [mt for mt in mcp_tools]
        if web_search_tool:
            tools.append(web_search_tool)

        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14",
                        http_client=httpx_client,
                        http_async_client=httpx_async_client,)
        llm_with_tools = llm.bind_tools(tools)

        def call_llm(state: AgentState):
            messages = state["messages"]
            assert isinstance(messages, list), "messages is not a list"
            assert len(messages) > 0, "messages is empty"
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_llm)
        workflow.add_node("tools", ToolNode(tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def invoke (self, state: AgentState) -> dict:
        full_messages_state = asyncio.run(self.graph.ainvoke(state))
        full_messages = full_messages_state["messages"]
        assert isinstance(full_messages, list) and len(full_messages) > len(state["messages"]), "Invalid full messages"

        new_messages = full_messages[len(state["messages"]):]
        new_messages = [
            m if isinstance(m, dict) else m.model_dump()
            for m in new_messages]
        return {
            "messages": new_messages,}

generalist_agent = GeneralistAgent()
