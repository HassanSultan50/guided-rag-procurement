import os
import time
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingest_live import fetch_live_rfx_data, parse_rfx_to_documents

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
BASE_URL = os.getenv("RFX_API_BASE_URL")
API_KEY = os.getenv("RFX_API_SUBSCRIPTION_KEY")
DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Throttle delay between API calls (seconds) to respect APIM rate limits
REQUEST_DELAY = 1.0


def fetch_all_rfx_ids():
    """Fetches the list of all available RFX IDs from the RFX API."""
    url = BASE_URL  # GET on the collection root returns all RFXs
    headers = {
        'Ocp-Apim-Subscription-Key': API_KEY,
        'Cache-Control': 'no-cache'
    }

    print(f"Fetching RFX list from: {url}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        rfx_list = response.json()
        # Adapt based on actual response shape:
        # If the API returns a list of objects with "id" fields:
        if isinstance(rfx_list, list):
            ids = [str(rfx.get("id") or rfx.get("Id") or rfx.get("rfxId")) for rfx in rfx_list]
        # If the API returns a wrapper object with a list inside:
        elif isinstance(rfx_list, dict):
            items = rfx_list.get("items") or rfx_list.get("data") or rfx_list.get("rfxs") or []
            ids = [str(item.get("id") or item.get("Id") or item.get("rfxId")) for item in items]
        else:
            ids = []

        ids = [rid for rid in ids if rid and rid != "None"]
        print(f"Found {len(ids)} RFX(s) available for ingestion.")
        return ids
    else:
        try:
            error_msg = response.json().get('message', response.text)
        except Exception:
            error_msg = response.text
        print(f"Failed to fetch RFX list. Status Code: {response.status_code}")
        print(f"Error: {error_msg}")
        response.raise_for_status()


def run_bulk_ingestion(rfx_ids=None):
    """
    Bulk ingestion pipeline.
    If rfx_ids is None, fetches all available RFXs from the API.
    If rfx_ids is a list, ingests only the specified RFXs.
    """
    if rfx_ids is None:
        rfx_ids = fetch_all_rfx_ids()

    if not rfx_ids:
        print("No RFX IDs to process. Exiting.")
        return

    all_documents = []
    succeeded = []
    failed = []

    print(f"\n{'='*50}")
    print(f"BULK INGESTION: Processing {len(rfx_ids)} RFX(s)")
    print(f"{'='*50}\n")

    for i, rfx_id in enumerate(rfx_ids, start=1):
        print(f"[{i}/{len(rfx_ids)}] Processing RFX: {rfx_id}")
        try:
            raw_json = fetch_live_rfx_data(rfx_id)
            docs = parse_rfx_to_documents(raw_json)
            print(f"  -> Parsed {len(docs)} document chunks.")
            all_documents.extend(docs)
            succeeded.append(rfx_id)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed.append(rfx_id)

        # Throttle to avoid APIM rate limits
        if i < len(rfx_ids):
            time.sleep(REQUEST_DELAY)

    if not all_documents:
        print("\nNo documents were parsed. Nothing to embed.")
        return

    # Embed and store all documents in a single batch
    print(f"\nEmbedding {len(all_documents)} total chunks into Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"BULK INGESTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Documents : {len(all_documents)} chunks embedded")
    print(f"  Database  : {DB_DIR}")
    if failed:
        print(f"  Failed IDs: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procurement RAG - Bulk RFX Ingestion")
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Optional list of specific RFX IDs to ingest. If omitted, fetches all from the API."
    )
    args = parser.parse_args()

    run_bulk_ingestion(rfx_ids=args.ids)