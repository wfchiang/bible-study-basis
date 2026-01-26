import copy
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .definitions import BibleBook, TextChunk
from .utils import extract_verses_text, make_bible_quote


def create_by_length_splitter(
        chunk_size: int = 500, overlap: int = 50,
        separators: List[str] | None = None) -> RecursiveCharacterTextSplitter:
    separators = separators or ["\n\n", "\n", "。", "？", "！", "；", "，", ",", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False)

def create_markdown_header_splitter(
        headers_to_split_on: list[tuple[str, str]] | None = None) -> MarkdownHeaderTextSplitter:
    headers_to_split_on = headers_to_split_on or [
        ("#", "title"),
        ("##", "section"),]
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)


def split_bible_book(
        bible_book: BibleBook,
        chunk_size: int = 400, overlap: int = 50) -> list[TextChunk]:
    chunks = []
    verse_cache = []
    for verse in bible_book.verses:
        cached_text = extract_verses_text(verse_cache)
        proposed_chunk_text = cached_text + verse.text
        if len(proposed_chunk_text) > chunk_size and verse_cache:
            bible_quote = make_bible_quote(
                book=bible_book.book, verses=verse_cache)
            bible_quote.metadata["category"] = "bible"
            chunks.append(bible_quote)
            while len(verse_cache) > 0 and len(extract_verses_text(verse_cache)) > overlap:
                verse_cache.pop(0)
        verse_cache.append(verse)
    if verse_cache:
        bible_quote = make_bible_quote(
            book=bible_book.book, verses=verse_cache)
        bible_quote.metadata["category"] = "bible"
        chunks.append(bible_quote)
    return chunks


def split_markdown_article(
        text: str, metadata: dict,
        chunk_size: int = 500, overlap: int = 100) -> list[TextChunk]:
    assert isinstance(metadata, dict), "Metadata must be a dictionary"

    md_splitter = create_markdown_header_splitter()
    len_splitter = create_by_length_splitter(
        chunk_size=chunk_size,
        overlap=overlap)
    
    md_sections = md_splitter.split_text(text)
    chunks = []
    for md_sec in md_sections:
        chunk_metadata = copy.deepcopy(metadata)
        md_sec_text = md_sec.page_content
        for k, v in md_sec.metadata.items():
            chunk_metadata[f"md_{k}"] = v

        md_sec_chunks = len_splitter.split_text(md_sec_text)
        for i, chunk_text in enumerate(md_sec_chunks):
            chunk = TextChunk(
                text=chunk_text,
                metadata={**chunk_metadata,
                          "category": "article",
                          "chunk_index": i})
            chunks.append(chunk)
    
    return chunks
