"""
Re-ingestion orchestrator — handles RFX update events.

Flow:
  1. Fetch current documents from Cosmos DB (for diff)
  2. Fetch latest RFX data from RFX API
  3. Parse into new document set
  4. Compute changelog (diff old vs new)
  5. Delete old documents from Cosmos DB
  6. Ingest new documents (embed + store)
  7. Store changelog entry
"""
import logging

from langfuse import observe
from src.ingest_live import fetch_live_rfx_data, parse_rfx_to_documents
from src.ingestion import get_rfx_documents, delete_rfx_documents, ingest_documents
from src.changelog import compute_changes, store_changelog

logger = logging.getLogger(__name__)


@observe(name="reingest_rfx")
def reingest_rfx(rfx_id: str) -> dict:
    """
    Performs a full re-ingestion of an RFX:
    - Deletes stale embeddings
    - Fetches and parses the latest version
    - Computes and stores a changelog
    - Re-embeds and stores the new documents

    Returns a summary dict with stats about what changed.
    """
    rfx_id = str(rfx_id)
    logger.info("[RFX %s] Starting re-ingestion...", rfx_id)

    # Step 1: Get existing documents from Cosmos (for changelog diff)
    old_documents = get_rfx_documents(rfx_id)
    logger.info("[RFX %s] Found %d existing documents.", rfx_id, len(old_documents))

    # Step 2: Fetch latest RFX data from RFX API
    raw_data = fetch_live_rfx_data(rfx_id)
    rfx_name = raw_data.get("name", "Unknown RFX")
    logger.info("[RFX %s] Fetched latest data: '%s'", rfx_id, rfx_name)

    # Step 3: Parse into new document set
    new_documents = parse_rfx_to_documents(raw_data)
    logger.info("[RFX %s] Parsed %d new documents.", rfx_id, len(new_documents))

    # Step 4: Compute changelog (before we delete old docs)
    if old_documents:
        changes = compute_changes(old_documents, new_documents)
        changelog_id = store_changelog(rfx_id, rfx_name, changes)
        logger.info("[RFX %s] Changelog stored: %s", rfx_id, changelog_id)
    else:
        changes = {
            "added": [], "removed": [], "modified": [],
            "stats": {
                "old_doc_count": 0,
                "new_doc_count": len(new_documents),
                "added_count": 0, "removed_count": 0, "modified_count": 0,
            },
        }
        logger.info("[RFX %s] First ingestion — no changelog needed.", rfx_id)

    # Step 5: Delete old documents
    deleted_count = delete_rfx_documents(rfx_id)
    logger.info("[RFX %s] Deleted %d old documents.", rfx_id, deleted_count)

    # Step 6: Ingest new documents (embed + store)
    ingest_documents(new_documents)
    logger.info("[RFX %s] Re-ingestion complete. %d documents stored.", rfx_id, len(new_documents))

    return {
        "rfx_id": rfx_id,
        "rfx_name": rfx_name,
        "old_doc_count": len(old_documents),
        "new_doc_count": len(new_documents),
        "deleted_count": deleted_count,
        "changes": changes["stats"],
    }
