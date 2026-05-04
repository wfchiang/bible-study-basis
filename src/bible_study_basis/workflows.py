import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import config


def rank_docs(query: str, docs: List[str]) -> Dict[str, Any]:
    """
    Reranks the relevance of the docs against the query using the LLM.

    Args:
        query: The search query.
        docs: A list of document strings.

    Returns:
        A dictionary containing the ranked documents with scores (0-5).
        Format:
        {
            "ranked": [
                {"index": <int>, "score": <0-5>},
                ...
            ]
        }
    """
    if len(docs) == 0:
        return {"ranked": []}

    # Initialize LLM with JSON mode enabled for structured output
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Format documents with IDs
    docs_input = "\n\n".join([f"ID {i}:\n{doc}" for i, doc in enumerate(docs)])

    prompt = f"""
You are a relevance ranking assistant.

Query: "{query}"

Documents:
{docs_input}

Task:
1. Evaluate the relevance of each document to the query.
2. Assign a score from 0 to 5 (0 = very irrelevant, 5 = very relevant).
3. Sort the results by score in descending order (most relevant first).
4. Return the result in the following JSON format:
{{
    "ranked": [
        {{ "index": <ID>, "score": <score> }},
        ...
    ]
}}
"""
    messages = [
        SystemMessage(content="You are a helpful assistant. You output valid JSON."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    
    try:
        result = json.loads(response.content)
        if not isinstance(result, dict) or "ranked" not in result:
            raise ValueError("Invalid response format")

        # Sort by score descending to ensure "rerank" behavior
        result["ranked"] = sorted(
            result["ranked"], 
            key=lambda x: float(x.get("score", 0)), 
            reverse=True)
        return result

    except Exception as e:
        return {"error": str(e)}
