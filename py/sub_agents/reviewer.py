import copy
import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from config import create_llm_with_tools
from data.definitions import AgentState, PROFESSION_OF_FAITH, BSBAgent


logger = logging.getLogger(__name__)


class ReviewFeedback(BaseModel):
    comment: str = Field(default="對於回答的評語。如果不批准，請具體說明哪些地方需要改進。")
    is_approved: bool = Field(default=False, description="是否批准這個回答。如果不批准，請在 comment 中說明原因。")


class ReviewerAgent(BSBAgent):
    system_message = f"""你需要為聖經學習的品質把關。
你的信仰宣言如下：
{PROFESSION_OF_FAITH}

你的審核標準如下：
1. 你的審核必須本著你的信仰宣言。你的審核必須確保回答是以聖經為根基的。
2. 你可以查找資料如解經文章等來幫助你理出聖經內容的脈絡，但你必須確保你的審核是以聖經為根基的。
3. 你必須確保回答要有聖經依據。這些聖經依據的內容必須主動提供。
4. 如果使用者的請求與聖經無關，請禮貌地告知使用者你只能回答與聖經相關的問題。

你的回答
"""

    async def create_graph(self) -> StateGraph:
        llm, tools = await create_llm_with_tools()
        llm = llm.with_structured_output(ReviewFeedback)

        async def call_llm(state: AgentState):
            messages = [SystemMessage(content=self.system_message)] + copy.deepcopy(state["messages"])
            assert isinstance(messages, list), "messages is not a list"
            assert len(messages) > 0, "messages is empty"

            response = await llm.ainvoke(messages)
            assert isinstance(response, ReviewFeedback), "response is not ReviewFeedback"

            if response.is_approved:
                logger.info("The answer is approved. comment: %s", response.comment)
                reviewer_messages = []
            else:
                logger.info("The answer is rejected. comment: %s", response.comment)
                reviewer_messages = [
                    AIMessage(content=f"Please refine the answer based on the following reviewer comment: {response.comment}")]
            return {
                "is_approved": response.is_approved,
                "messages": reviewer_messages,
                "n_pushbacks": state.get("n_pushbacks", 0) + (0 if response.is_approved else 1),
            }

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
