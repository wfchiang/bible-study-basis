import json
import logging
from pathlib import Path

import click
from tqdm import tqdm

from data.definitions import TextChunk
from db.vector_store import add_text_chunk, create_collection_if_not_exists

DEFAULT_DATA_FILE = Path(__file__).parents[1] / "build" / "dataset.jsonl"

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("vector_store").setLevel(logging.ERROR)


@click.command()
@click.option("--create-collection", is_flag=True, help="Create the collection if it does not exist.")
@click.argument("data_file", default=DEFAULT_DATA_FILE, type=click.Path())
def main(
    create_collection: bool,
    data_file: Path
):
    # Create the missed collection
    if create_collection:
        create_collection_if_not_exists()

    total_lines = sum(1 for _ in open(data_file, "rb"))

    with data_file.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Publishing data"):
            data = json.loads(line)
            b_chunk = TextChunk(**data)
            add_text_chunk(b_chunk)


if __name__ == "__main__":
    main()
