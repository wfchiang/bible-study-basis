import os
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from data.definitions import AgentState
from sub_agents.generalist import GeneralistAgent
from sub_agents.reviewer import ReviewerAgent


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def check_env_vars() -> None:
    """
    Checks if the required environment variables are set.
    Raises an exception if any are missing.
    """
    required_vars = ["BSB_MCP_SERVER", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")


async def create_agent() -> StateGraph:
    check_env_vars()

    generalist_agent = await GeneralistAgent().create_graph()
    reviewer_agent = await ReviewerAgent().create_graph()

    def should_continue(state: AgentState):
        if state.get("is_approved", False):
            return "postproc"
        if state.get("n_pushbacks", 0) >= 3:
            logger.warning("The answer has been rejected 3 times. Stopping further attempts.")
            return "postproc"
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", generalist_agent.ainvoke)
    workflow.add_node("reviewer", reviewer_agent.ainvoke)
    workflow.add_node("postproc", postproc)
    

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "reviewer")
    workflow.add_conditional_edges("reviewer", should_continue)
    workflow.add_edge("postproc", END)

    return workflow.compile()


async def postproc(state: AgentState) -> dict:
    """
    Post-process the messages to extract the final answer.
    """
    for i, m in enumerate(state["messages"]):
        if (isinstance(m, AIMessage) or isinstance(m, HumanMessage)
                or isinstance(m, SystemMessage) or isinstance(m, ToolMessage)):
            state["messages"][i] = m.model_dump()
    return {}

if __name__ == "__main__":
    import asyncio
    async def main():
        agent = await create_agent()

        result = await agent.ainvoke({
            # "is_approved": False,
            # "n_pushbacks": 0,
            "messages": [
                # {"role": "user", "content": "遵守神的命令會帶來真正的喜樂嗎?為什麼?"}
                # {"role": "user", "content": "請透過搜尋 'article' 來找到關於啟示綠主旨的解釋"}
                {"role": "user", "content": "明年總統大選誰會贏?"}
            ]
        })
        print(type(result))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    asyncio.run(main())
