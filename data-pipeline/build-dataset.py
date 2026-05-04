import io
import json
import logging
from pathlib import Path
import tempfile

import click
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken
import yaml

from data.definitions import Bible, BibleBook
from data.definitions import TextChunk
from data.loaders import load_bible_from_dir
from data.utils import create_markdown_from_TextChunk

DEFAULT_CHUNK_SIZE = 7500
DEFAULT_DATA_DIR = Path(__file__).parents[1] / "data"
DEFAULT_OUTPUT_FILE = Path(__file__).parents[1] / "build" / "dataset.jsonl"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DOCLING_CONVERTER = DocumentConverter()
OPENAI_TIKTOKEN_TOKENIZER = tiktoken.encoding_for_model("text-embedding-3-small")


def chunk_docling_doc(
        doc, chunk_size: int, context_creation_fn):
    tokenizer = OpenAITokenizer(
        tokenizer=OPENAI_TIKTOKEN_TOKENIZER,
        max_tokens=chunk_size,)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True)

    raw_chunks = chunker.chunk(doc)
    chunks = []
    for chunk in raw_chunks:
        context = context_creation_fn(chunk)
        chunks.append(TextChunk(
            text=f"<context>\n{context}\n</context>\n\n<content>\n{chunk.text}\n</content>",
            metadata={
                "context": context,
            }
        ))
    return chunks\


def create_chunk_context(chunk) -> str:
    """Creates a contextualized chunk with metadata."""
    context = " > ".join([str(h) for h in chunk.meta.headings]) if chunk.meta.headings else "General"
    return context


def create_chunks_from_bible(bible: Bible, chunk_size: int = DEFAULT_CHUNK_SIZE):
    """Chunks a BibleBook and yields contextualized chunks."""
    logger.info(f"Processing and chunking {bible.version}...")
    doc_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            bible_md = create_markdown_from_TextChunk(bible)
            f.write(bible_md)
            temp_path = Path(f.name)
            logger.info(f"Written Bible version {bible.version} to temporary file {temp_path}")
            chunks = create_chunks_from_doc_path(temp_path, chunk_size)
    finally:
        if doc_path and doc_path.exists():
            doc_path.unlink()
    return chunks


def create_chunks_from_doc_path(doc_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE):
    """Chunks a document and yields contextualized chunks."""
    logger.info(f"Processing and chunking {doc_path}...")
    result = DOCLING_CONVERTER.convert(str(doc_path))
    doc = result.document
    return chunk_docling_doc(
        doc, chunk_size, 
        context_creation_fn=create_chunk_context)


@click.command()
@click.option(
    "--data-dir",
    default=DEFAULT_DATA_DIR,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing the data files.",
)
@click.option(
    "--output",
    "output_file",
    default=DEFAULT_OUTPUT_FILE,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the output JSONL file.",
    show_default=True,
)
def main(data_dir: Path, output_file: Path):
    """
    This script builds the dataset from the data directory by chunking documents.
    """
    logger.info(f"Using data directory: {data_dir}")

    # Create the output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    chunks = []

    # Search for Bible versions and create chunks
    bible_versions_dir = data_dir / "bible_versions"
    for bible_ver_dir in bible_versions_dir.glob("*"):
        if not bible_ver_dir.is_dir():
            continue
        logger.info(f"Processing Bible version: {bible_ver_dir.name}")
        bible = load_bible_from_dir(bible_ver_dir)

        bible_chunks = create_chunks_from_bible(bible)
        for b_chunk in bible_chunks:
            b_chunk.metadata["category"] = "bible"
            b_chunk.metadata["bible_version"] = bible.version
        chunks.extend(bible_chunks)

    # Search for articles and create chunks
    articles_dir = data_dir / "articles"
    article_index_file = articles_dir / "index.yaml"
    with open(article_index_file, "r", encoding="utf-8") as f:
        article_index = yaml.safe_load(f)
        for article_idx in article_index["articles"]:
            article_title = article_idx["title"]
            article_path = articles_dir / article_idx["file"]
            assert article_path.exists(), f"Article file {article_path} does not exist"
            
            logger.info(f"Processing article: {article_title} from {article_path}")
            article_chunks = create_chunks_from_doc_path(article_path)
            for a_chunk in article_chunks:
                a_chunk.metadata["category"] = "article"
            chunks.extend(article_chunks)

    # Export the chunks to a JSONL file
    with output_file.open("w", encoding="utf-8") as output_f:
        for chunk in chunks:
            output_f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    logger.info(f"Dataset build complete. Total chunks created: {len(chunks)}")
    logger.info(f"Output written to {output_file}")

if __name__ == "__main__":
    main()