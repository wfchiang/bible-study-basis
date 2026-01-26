from datetime import datetime
import json
import logging
from pathlib import Path

import click
import yaml

from config import config
from data.definitions import Bible, TextChunk
from data.loaders import load_bible_from_dir
from data.splitters import split_bible_book, split_markdown_article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

DEFAULT_DATA_FILE = Path(__file__).parents[1] / "build" / "data.jsonl"

_now_time = datetime.now()
_data_build_timestamp = _now_time.strftime("%Y%m%d%H%M%S")
_data_build_index = 0
def get_data_build_id() -> str:
    global _data_build_index
    data_build_id = f"{_data_build_timestamp}-{_data_build_index}"
    _data_build_index += 1
    return data_build_id


@click.command()
@click.argument("data_file", default=DEFAULT_DATA_FILE, type=click.Path())
def main(
    data_file: Path
):
    data_file.parent.mkdir(parents=True, exist_ok=True)
    with data_file.open("w", encoding="utf-8") as f:
        # Load Bible versions
        for bible_ver_dir in config["data"]["bible_versions"]:
            bible_ver_path = Path(bible_ver_dir)
            assert bible_ver_path.is_dir()

            logger.info("Loading a Bible version from %s", bible_ver_dir)
            bible_ver = load_bible_from_dir(bible_ver_path)
            assert isinstance(bible_ver, Bible)

            for bible_book in bible_ver.books:
                logging.info("-- Processing the book of %s", bible_book.book)
                book_chunks = split_bible_book(bible_book)
                for b_chunk in book_chunks:
                    assert isinstance(b_chunk, TextChunk)
                    b_chunk.metadata["data_build_id"] = get_data_build_id()
                    f.write(json.dumps(b_chunk.model_dump(), ensure_ascii=False) + "\n")
        
        # Load articles
        article_base_dir = Path(__file__).parents[1] / "data" / "articles"
        article_index_file = article_base_dir / "index.yaml"
        if article_index_file.is_file():
            with article_index_file.open("r", encoding="utf-8") as idx_f:
                article_index = yaml.safe_load(idx_f)
            for article_metadata in article_index.get("articles", []):
                if not isinstance(article_metadata, dict) or "file" not in article_metadata:
                    continue
                article_file = article_base_dir / article_metadata["file"]
                if not article_file.is_file():
                    logger.warning("Article file %s not found, skipping.", article_file)
                    continue

                logger.info("Processing article file %s", article_file)
                article_chunks = split_markdown_article(
                    text=article_file.read_text(encoding="utf-8"),
                    metadata=article_metadata)
                for art_chunk in article_chunks:
                    assert isinstance(art_chunk, TextChunk)
                    art_chunk.metadata["data_build_id"] = get_data_build_id()
                    f.write(json.dumps(art_chunk.model_dump(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
