import os
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from data.definitions import AgentState
from sub_agents.generalist import GeneralistAgent
from sub_agents.planner import PlannerAgent
from sub_agents.qa import QAAgent


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
    # planner_agent = PlannerAgent()
    # qa_agent = await QAAgent().create_graph()

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", generalist_agent.ainvoke)
    workflow.add_node("postproc", postproc)
    # workflow.add_node("qa", qa_agent.invoke)

    workflow.add_edge(START, "agent")
    # workflow.add_edge(START, "qa")
    workflow.add_edge("agent", "postproc")
    workflow.add_edge("postproc", END)
    # workflow.add_edge("qa", END)

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
            "messages": [
                # {"role": "user", "content": "遵守神的命令會帶來真正的喜樂嗎?為什麼?"}
                {"role": "user", "content": "請透過搜尋 'article' 來找到關於啟示綠主旨的解釋"}
            ]
        })
        print(type(result))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    asyncio.run(main())
