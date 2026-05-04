import asyncio
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from config import httpx_client, httpx_async_client, mcp_client
from data.definitions import PROFESSION_OF_FAITH, AgentState, BSBAgent


class PlannerAgent(BSBAgent):
    prompt_template = """你是一個基督徒聖經學者。你的信仰如下：
{profession_of_faith}

你可以提供以下服務項目 (服務名稱: 描述)：
{service_items}

你即將要處裡一個使用者的請求。請根據你的信仰和服務項目，提出一個詳細的計畫。
你的回應將會是一個以 JSON 物件描述的 DAG (directed acyclic graph)，其中包含一個或多個節點，每個節點代表一個任務或步驟。
每個節點應包含以下欄位：
- id: 任務的唯一識別碼
- service: 服務名稱 (必須是你能提供的服務項目之一)
- input: 服務的輸入參數 (str)，請提取部分的 (或全部) 的使用者請求，並以最少的修改做成一個輸入參數
- children: 子任務的 id 陣列 (可以是空陣列)

請確保你的回應是有效的 JSON 格式，並且符合以下結構。

{{
    "root": "<root_node_id>",
    "nodes": {{
        "<node_id>": {{
            "id": <node_id>,
            "service": <service_name>,
            "input": <service_input>,
            "children": [<child_node_id_1>, <child_node_id_2>, ...]
        }},
        ...
    }}
}}

請注意：
- 有可能使用者的請求並不複雜，因此計畫可能只有一個節點。

現在，請根據以下使用者的請求，生成計畫。
使用者請求：{user_request}
"""

    services = {
        "question_answering": "回答有關聖經內容的問題",
        "small_group_discussion": "提供小組討論的指引和問題",
        "misc": "其他相關 (或非相關) 的聖經學習服務",
    }

    def _create_prompt(self, user_request: str) -> str:
        return self.prompt_template.format(
            profession_of_faith=PROFESSION_OF_FAITH,
            service_items=self._encode_service_items(),
            user_request=user_request,)


    def _encode_service_items(self) -> str:
        items = []
        for k in self.services.keys():
            items.append(f"- {k}: {self.services[k]}")
        return "\n".join(items)


    async def _create_graph(self) -> StateGraph:
        mcp_tools = await mcp_client.get_tools()
        tools = mcp_tools

        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14",
                        http_client=httpx_client,
                        http_async_client=httpx_async_client,)
        llm_with_tools = llm.bind_tools(tools)

        def call_llm(state: AgentState):
            response = llm_with_tools.invoke(state["messages"])
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
        """
        Create a planner node for building the multi-agent system.
        """
        last_message = state["messages"][-1]
        user_request = last_message["content"]

        plan = asyncio.run(self.make_plan(user_request))
        return {
            "plan": plan,
        }


    async def make_plan(self, user_request: str) -> dict:
        """
        According to the user's request, generate a plan.
        The plan should be a JSON object describing a directed acyclic graph (DAG),
        """
        prompt = self._create_prompt(user_request)
        response = await self.graph.ainvoke({
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
        assert isinstance(response, dict), f"Invalid response type: {type(response)}"
        last_message: AIMessage = response["messages"][-1]
        assert isinstance(last_message, AIMessage), "Last message is not an AIMessage"

        plan = json.loads(last_message.content)
        return plan


planner_agent = PlannerAgent()
