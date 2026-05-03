import logging
import uuid
import os
from pathlib import Path

from qdrant_client import QdrantClient, models, grpc

from data.definitions import TextChunk
from config import config, embedding_length, embedding_model


logger = logging.getLogger("vector_store")

vs_config = config["vector_store"]
vs_collection_name = vs_config["collection_name"]

storage_dir = Path(__file__).resolve().parents[2] / "vector_store_data"

# --- Global instances (initialized once for efficiency) ---
# Initialize the Qdrant client.
qdrant_client = QdrantClient(path=storage_dir)


def create_collection_if_not_exists() -> None:
    """
    Creates the Qdrant collection if it does not already exist.
    """
    try:
        collection_response: grpc.GetCollectionInfoResponse = qdrant_client.get_collection(collection_name=vs_collection_name)
        collection_status = str(collection_response.status).lower()
        if collection_status == "green":
            logger.info("Collection '%s' already exists.", vs_collection_name)
            return
        raise RuntimeError(f"Unrecognized collection status: {collection_status}")
    except Exception as e:
        # Assuming an exception means the collection does not exist.
        # A more specific exception type could be caught if the client library provides it.
        logger.error("Error checking collection existence: assuming it does not exist.", e, exc_info=True)

    logger.info("Collection '%s' does not exist. Creating it...", vs_collection_name)
    qdrant_client.create_collection(
        collection_name=vs_collection_name,
        vectors_config=models.VectorParams(
            size=embedding_length,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("Collection '%s' created successfully.", vs_collection_name)


def add_text_chunk(text_chunk: TextChunk) -> None:
    """
    Generates an embedding for a TextChunk and upserts it with its metadata
    into the Qdrant collection.
    """
    vector = embedding_model.embed_documents([text_chunk.text])[0]
    payload = {"text": text_chunk.text, **text_chunk.metadata}
    point_id = str(uuid.uuid4())

    qdrant_client.upsert(
        collection_name=vs_collection_name,
        points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
        wait=True,
    )
    logger.info("Successfully added chunk with id %s to Qdrant collection '%s'.", point_id, vs_collection_name)


def search_text_chunks(text: str, top_k: int = 30, filters: dict | None = None) -> list[dict]:
    vector = embedding_model.embed_documents([text])[0]

    query_filter = None
    if filters:
        conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
        query_filter = models.Filter(must=conditions)

    search_results = qdrant_client.query_points(
        collection_name=vs_collection_name,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True, with_vectors=False)
    points_payload = [
        srel.payload
        for srel in search_results.points]
    assert all(isinstance(pp, dict) for pp in points_payload)
    return points_payload
