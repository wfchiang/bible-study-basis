import os
import json

from langgraph.graph import StateGraph, START, END

from data.definitions import AgentState, PROFESSION_OF_FAITH
from sub_agents.generalist import generalist_agent
from sub_agents.planner import planner_agent


# 1. Global variable for the system prompt. You can edit this!
SYSTEM_PROMPT = """你是一位聖經學習助手，專注於幫助使用者理解和學習聖經內容。請根據使用者的問題，本著聖經的信息來回到以聖經為出發點的答案。
你的信仰宣言如下：
{profession_of_faith}
""".format(
    profession_of_faith=PROFESSION_OF_FAITH)


services: dict[str, dict] = {
    "question_answering": {
        "description": "回答有關聖經內容的問題",
        "agent": generalist_agent
    },
    "small_group_discussion": {
        "description": "提供小組討論的指引和問題",
        "agent": generalist_agent
    },
    "misc": {
        "description": "其他相關 (或非相關) 的聖經學習服務",
        "agent": generalist_agent
    },
}


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

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_agent.invoke)
    workflow.add_node("agent", generalist_agent.invoke)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


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
