import time
import logging
import requests
from src.config import RFX_API_BASE_URL, RFX_API_SUBSCRIPTION_KEY
from src.ingest_live import fetch_live_rfx_data, parse_rfx_to_documents
from src.ingestion import ingest_documents

logger = logging.getLogger(__name__)

REQUEST_DELAY = 1.0


def fetch_all_rfx_ids():
    """Fetches the list of all available RFX IDs from the RFX API."""
    url = RFX_API_BASE_URL
    headers = {
        'Ocp-Apim-Subscription-Key': RFX_API_SUBSCRIPTION_KEY,
        'Cache-Control': 'no-cache'
    }

    logger.info("Fetching RFX list from: %s", url)
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        rfx_list = response.json()
        if isinstance(rfx_list, list):
            ids = [str(rfx.get("id") or rfx.get("Id") or rfx.get("rfxId")) for rfx in rfx_list]
        elif isinstance(rfx_list, dict):
            items = rfx_list.get("items") or rfx_list.get("data") or rfx_list.get("rfxs") or []
            ids = [str(item.get("id") or item.get("Id") or item.get("rfxId")) for item in items]
        else:
            ids = []

        ids = [rid for rid in ids if rid and rid != "None"]
        logger.info("Found %d RFX(s) available for ingestion.", len(ids))
        return ids
    else:
        try:
            error_msg = response.json().get('message', response.text)
        except Exception:
            error_msg = response.text
        logger.error("Failed to fetch RFX list. Status: %d, Error: %s", response.status_code, error_msg)
        response.raise_for_status()


def run_bulk_ingestion(rfx_ids=None):
    """Bulk ingestion pipeline using Cosmos DB."""
    if rfx_ids is None:
        rfx_ids = fetch_all_rfx_ids()

    if not rfx_ids:
        logger.info("No RFX IDs to process. Exiting.")
        return

    all_documents = []
    succeeded = []
    failed = []

    logger.info("BULK INGESTION: Processing %d RFX(s)", len(rfx_ids))

    for i, rfx_id in enumerate(rfx_ids, start=1):
        logger.info("[%d/%d] Processing RFX: %s", i, len(rfx_ids), rfx_id)
        try:
            raw_json = fetch_live_rfx_data(rfx_id)
            docs = parse_rfx_to_documents(raw_json)
            logger.info("  -> Parsed %d document chunks.", len(docs))
            all_documents.extend(docs)
            succeeded.append(rfx_id)
        except Exception as e:
            logger.error("  -> FAILED: %s", e)
            failed.append(rfx_id)

        if i < len(rfx_ids):
            time.sleep(REQUEST_DELAY)

    if not all_documents:
        logger.info("No documents were parsed. Nothing to embed.")
        return

    logger.info("Embedding %d total chunks into Cosmos DB...", len(all_documents))
    ingest_documents(all_documents)

    logger.info("BULK INGESTION COMPLETE — Succeeded: %d, Failed: %d, Documents: %d",
                len(succeeded), len(failed), len(all_documents))
    if failed:
        logger.warning("Failed IDs: %s", ", ".join(failed))
