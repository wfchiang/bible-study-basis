import logging
from pathlib import Path

import fastmcp

from config import config
from data.definitions import BibleBook
from data.utils import make_bible_quote
from data.loaders import load_bible_from_dir
from db.vector_store import search_text_chunks
from workflows import rank_docs


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp_app = fastmcp.FastMCP("Bible-study-bot MCP")

bible_versions = {}
for bible_version_path in config["data"]["bible_versions"]:
    bible_version = load_bible_from_dir(
        Path(bible_version_path))
    bible_versions[bible_version.version] = bible_version
assert len(bible_versions) > 0


@mcp_app.tool(
        name="get_bible_verses",
        description="提取特定的圣经经文或经文范围。")
async def get_bible_verses(
        book: str,
        from_chapter: int,
        from_verse: int,
        to_chapter: int | None = None,
        to_verse: int | None = None,
        version: str | None = None) -> dict:
    try:
        book = book.lower()
        to_chapter = to_chapter or from_chapter
        to_verse = to_verse or from_verse
        logger.info("Getting Bible verses: %s %d:%d to %d:%d (version: %s)",
                    book, from_chapter, from_verse, to_chapter, to_verse,
                    version or "default")

        assert 1 <= from_chapter <= to_chapter
        assert 1 <= from_verse and 1 <= to_verse
        if from_chapter == to_chapter:
            assert from_verse <= to_verse

        version = version or list(bible_versions.keys())[0]
        bible_version = bible_versions[version]

        bible_book = None
        for bb in bible_version.books:
            if bb.book == book:
                bible_book = bb
                break
        assert isinstance(bible_book, BibleBook), "Invalid book name. Valid names are: " + ", ".join([bb.book for bb in bible_version.books])

        got_from_v = False
        got_to_v = False
        verses = []
        for bv in bible_book.verses:
            if not got_from_v:
                if bv.chapter == from_chapter and bv.verse == from_verse:
                    got_from_v = True
            if got_from_v:
                verses.append(bv)
            if not got_to_v:
                if bv.chapter == to_chapter and bv.verse == to_verse:
                    got_to_v = True
            if got_to_v:
                break
        assert got_from_v, "Invalid 'from' verse"
        assert got_to_v, "Invalid 'to' verse"

        bible_quote = make_bible_quote(
            book=book, verses=verses)
        return bible_quote.model_dump()

    except Exception as e:
        logger.error("Error in get_bible_verses: %s", e)
        return {
            "error": str(e)
        }


def _search_and_rank_chunks(
        query: str,
        category: str,
        min_score: float,
        top_k: int) -> list | dict:
    assert top_k > 0, "top_k must be greater than 0"
    logger.info("Searching %s chunks for query: %s", category, query)

    text_chunks = search_text_chunks(
        query,
        filters={"category": category},
        top_k=top_k * 2,)
    logger.info("Found %s text chunks", len(text_chunks))

    text_list = [tc["text"] for tc in text_chunks]
    ranked_docs = rank_docs(query=query, docs=text_list)
    if "error" in ranked_docs:
        logger.error("Error in ranking docs: %s", ranked_docs["error"])
        return {"error": ranked_docs["error"]}
    assert "ranked" in ranked_docs, "Invalid ranked docs format"

    logger.info("Ranking text chunks")
    ranked_chunks = []
    for rd in ranked_docs.get("ranked", []):
        try:
            index = int(rd.get("index"))
            score = float(rd.get("score", 0.0))
            if 0 <= index < len(text_chunks) and score >= min_score:
                ranked_chunks.append(text_chunks[index])
        except Exception as e:
            logger.error("Error in ranking docs: %s", e)

    if len(ranked_chunks) > top_k:
        ranked_chunks = ranked_chunks[:top_k]
    logger.info("Returning %s ranked chunks", len(ranked_chunks))
    return ranked_chunks


@mcp_app.tool(
        name="search_article_chunks",
        description="搜寻与查找相关的文章片段，以回答用户关于特定主题的问题。")
async def search_article_chunks(
        query: str,
        min_score: float = 3.0, top_k: int = 5) -> dict:
    """
    Search for article chunks relevant to the query.
    Use this tool to find relevant articles when the user asks about specific topics.
    """
    result = _search_and_rank_chunks(
        query=query, category="article", min_score=min_score, top_k=top_k)
    if isinstance(result, dict) and "error" in result:
        return result
    return {
        "article_chunks": result}


@mcp_app.tool(
        name="search_bible_chunks",
        description="搜寻与查找相关的圣经经文或文本片段，以回答用户关于特定主题、经文或神学概念的问题。")
async def search_bible_chunks(
        query: str,
        min_score: float = 3.0, top_k: int = 5) -> dict:
    """
    Search for Bible verses or text chunks relevant to the query.
    Use this tool to find relevant scripture when the user asks about specific topics, verses, or theological concepts in the Bible.
    """
    result = _search_and_rank_chunks(
        query=query, category="bible", min_score=min_score, top_k=top_k)
    if isinstance(result, dict) and "error" in result:
        return result
    return {
        "bible_chunks": result}


if __name__ == "__main__":
    # Use streamable-http transport for production
    mcp_app.run(
        transport=config["mcp"]["transport"],
        host="0.0.0.0", port=config["mcp"]["port"])
