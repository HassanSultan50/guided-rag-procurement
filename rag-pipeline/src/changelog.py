"""
Changelog module — tracks differences between RFX versions.

Compares old documents (from Cosmos DB) against newly parsed documents
and stores a structured changelog entry in the rfx-changelog container.
"""
import uuid
import logging
from datetime import datetime, timezone

from src.ingestion import ensure_changelog_container

logger = logging.getLogger(__name__)


def _build_document_map(documents: list[dict], key_field: str = "page_content") -> dict[str, dict]:
    """
    Builds a lookup map from documents keyed by (type, supplier, content_hash).
    For old docs (from Cosmos), items have 'page_content' and 'metadata' at top level.
    For new docs (Document objects), items have .page_content and .metadata attributes.
    """
    doc_map = {}
    for doc in documents:
        # Handle both dict (from Cosmos query) and Document objects
        if isinstance(doc, dict):
            meta = doc.get("metadata", {})
            content = doc.get("page_content", "")
        else:
            meta = doc.metadata
            content = doc.page_content

        doc_type = meta.get("type", "unknown")
        supplier = meta.get("supplier", "_global")
        # Key: combination of type + supplier ensures we compare apples to apples
        key = f"{doc_type}|{supplier}"

        if key not in doc_map:
            doc_map[key] = []
        doc_map[key].append(content)

    return doc_map


def compute_changes(old_documents: list[dict], new_documents: list) -> dict:
    """
    Computes a structured diff between old and new document sets.

    Returns a dict with:
    - added: list of {type, supplier, summary} for new content
    - removed: list of {type, supplier, summary} for deleted content
    - modified: list of {type, supplier, old_summary, new_summary} for changed content
    - stats: {old_count, new_count, added_count, removed_count, modified_count}
    """
    old_map = _build_document_map(old_documents)
    new_map = _build_document_map(new_documents)

    all_keys = set(list(old_map.keys()) + list(new_map.keys()))

    added = []
    removed = []
    modified = []

    for key in sorted(all_keys):
        doc_type, supplier = key.split("|", 1)
        old_contents = old_map.get(key, [])
        new_contents = new_map.get(key, [])

        old_set = set(old_contents)
        new_set = set(new_contents)

        # Completely new documents for this key
        if not old_contents and new_contents:
            added.append({
                "type": doc_type,
                "supplier": supplier,
                "summary": f"{len(new_contents)} new document(s)",
                "sample": new_contents[0][:200] if new_contents else "",
            })
            continue

        # Documents removed for this key
        if old_contents and not new_contents:
            removed.append({
                "type": doc_type,
                "supplier": supplier,
                "summary": f"{len(old_contents)} document(s) removed",
                "sample": old_contents[0][:200] if old_contents else "",
            })
            continue

        # Both exist — check for modifications
        content_added = new_set - old_set
        content_removed = old_set - new_set

        if content_added or content_removed:
            # Content changed for this type+supplier combination
            change_details = []
            if content_added:
                change_details.append(f"{len(content_added)} new/updated answer(s)")
            if content_removed:
                change_details.append(f"{len(content_removed)} previous answer(s) replaced")

            modified.append({
                "type": doc_type,
                "supplier": supplier,
                "summary": "; ".join(change_details),
                "old_sample": list(content_removed)[0][:200] if content_removed else "",
                "new_sample": list(content_added)[0][:200] if content_added else "",
            })

    changes = {
        "added": added,
        "removed": removed,
        "modified": modified,
        "stats": {
            "old_doc_count": len(old_documents),
            "new_doc_count": len(new_documents),
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
        },
    }

    return changes


def store_changelog(rfx_id: str, rfx_name: str, changes: dict) -> str:
    """
    Stores a changelog entry in the rfx-changelog Cosmos container.
    Returns the changelog entry ID.
    """
    container = ensure_changelog_container()

    entry_id = str(uuid.uuid4())
    entry = {
        "id": entry_id,
        "rfx_id": rfx_id,
        "rfx_name": rfx_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changes": changes,
        "has_changes": (
            changes["stats"]["added_count"] > 0
            or changes["stats"]["removed_count"] > 0
            or changes["stats"]["modified_count"] > 0
        ),
    }

    container.upsert_item(entry)
    logger.info(
        "Changelog stored for RFX %s: +%d -%d ~%d changes.",
        rfx_id,
        changes["stats"]["added_count"],
        changes["stats"]["removed_count"],
        changes["stats"]["modified_count"],
    )
    return entry_id
