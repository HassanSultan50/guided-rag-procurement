import uuid
import hashlib
import logging
from azure.cosmos import CosmosClient, PartitionKey
from langfuse import observe
from src.retriever import embed_documents
from src.config import (
    COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER,
    COSMOS_CHANGELOG_CONTAINER, VECTOR_DIMENSIONS,
)

logger = logging.getLogger(__name__)


def _deterministic_id(metadata: dict, content: str) -> str:
    """
    Generates a deterministic document ID from metadata + content hash.
    Format: {rfx_id}_{type}_{supplier}_{content_hash8}
    This ensures the same document always gets the same ID, enabling
    efficient upserts and avoiding duplicates on re-ingestion.
    """
    rfx_id = metadata.get("rfx_id", "unknown")
    doc_type = metadata.get("type", "unknown")
    supplier = metadata.get("supplier", "_global")
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{rfx_id}_{doc_type}_{supplier}_{content_hash}"


def ensure_cosmos_container():
    """
    Creates the Cosmos DB database and container with DiskANN vector index
    if they don't already exist.
    """
    client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    database = client.create_database_if_not_exists(id=COSMOS_DATABASE)

    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": VECTOR_DIMENSIONS,
            }
        ]
    }

    indexing_policy = {
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": "/embedding/*"}],
        "vectorIndexes": [
            {"path": "/embedding", "type": "diskANN"}
        ],
    }

    container = database.create_container_if_not_exists(
        id=COSMOS_CONTAINER,
        partition_key=PartitionKey(path="/metadata/rfx_id"),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
    )
    logger.info("Cosmos container '%s' ready.", COSMOS_CONTAINER)
    return container


@observe(name="ingest_documents")
def ingest_documents(documents, batch_size=50):
    """
    Embeds documents via AI Foundry and upserts into Cosmos DB.
    Replaces Chroma.from_documents().
    """
    container = ensure_cosmos_container()

    texts = [doc.page_content for doc in documents]
    all_embeddings = embed_documents(texts, batch_size=batch_size)

    for doc, embedding in zip(documents, all_embeddings):
        item = {
            "id": _deterministic_id(doc.metadata, doc.page_content),
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": embedding,
        }
        container.upsert_item(item)

    logger.info("Ingested %d documents into Cosmos DB.", len(documents))


def get_rfx_documents(rfx_id: str) -> list[dict]:
    """
    Retrieves all documents for a given RFX from Cosmos DB.
    Uses the partition key (metadata/rfx_id) for efficient reads.
    Returns raw items (without embeddings to save memory).
    """
    container = ensure_cosmos_container()
    query = "SELECT c.id, c.page_content, c.metadata FROM c WHERE c.metadata.rfx_id = @rfx_id"
    params = [{"name": "@rfx_id", "value": rfx_id}]

    items = list(container.query_items(
        query=query,
        parameters=params,
        partition_key=rfx_id,
    ))
    logger.info("Retrieved %d existing documents for RFX %s.", len(items), rfx_id)
    return items


def delete_rfx_documents(rfx_id: str) -> int:
    """
    Deletes all documents for a given RFX from Cosmos DB.
    Uses partition key for efficient deletion.
    Returns the number of documents deleted.
    """
    container = ensure_cosmos_container()

    # Query all document IDs in this partition
    query = "SELECT c.id FROM c WHERE c.metadata.rfx_id = @rfx_id"
    params = [{"name": "@rfx_id", "value": rfx_id}]

    items = list(container.query_items(
        query=query,
        parameters=params,
        partition_key=rfx_id,
    ))

    count = 0
    for item in items:
        container.delete_item(item=item["id"], partition_key=rfx_id)
        count += 1

    logger.info("Deleted %d documents for RFX %s.", count, rfx_id)
    return count


def ensure_changelog_container():
    """
    Creates the changelog container in Cosmos DB if it doesn't exist.
    No vector index needed — this is a simple document store.
    """
    client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    database = client.create_database_if_not_exists(id=COSMOS_DATABASE)

    container = database.create_container_if_not_exists(
        id=COSMOS_CHANGELOG_CONTAINER,
        partition_key=PartitionKey(path="/rfx_id"),
    )
    logger.info("Changelog container '%s' ready.", COSMOS_CHANGELOG_CONTAINER)
    return container
