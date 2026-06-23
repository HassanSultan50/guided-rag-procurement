"""
Full Data Refresh: Re-ingest all RFXs and regenerate ground truth from the SAME data.

This ensures ground truth is perfectly aligned with the indexed data — the #1 reason
experiment scores were low was a mismatch between these two datasets.

Usage:
    python refresh_data_and_ground_truth.py                # Full pipeline
    python refresh_data_and_ground_truth.py --skip-ingest   # Only regenerate ground truth
    python refresh_data_and_ground_truth.py --rfx-id 2100177 # Single RFX only
"""

import json
import os
import sys
import time
import logging
import argparse
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config  # noqa: F401
from src.ingest_live import fetch_live_rfx_data, parse_rfx_to_documents
from src.ingestion import delete_rfx_documents, ingest_documents
from generate_ground_truth import generate_qa_pairs, summarize_rfx_for_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truth_data")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "rfx_snapshots")

# All RFX IDs in the system
RFX_IDS = [
    "2100148", "2100149", "2100150", "2100151", "2100158", "2100159", "2100160",
    "2100161", "2100164", "2100165", "2100166", "2100167", "2100168", "2100169",
    "2100170", "2100171", "2100172", "2100175", "2100176", "2100177", "2100180",
    "2100181", "2100182",
]


def refresh_all(skip_ingest: bool = False, rfx_id_filter: str = None):
    """Re-ingest all RFXs and regenerate ground truth from the same API snapshot."""

    rfx_ids = [rfx_id_filter] if rfx_id_filter else RFX_IDS
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Backup existing ground truth
    if os.path.isdir(GROUND_TRUTH_DIR):
        backup_name = f"ground_truth_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(BASE_DIR, backup_name)
        shutil.copytree(GROUND_TRUTH_DIR, backup_path)
        logger.info("Backed up existing ground truth to: %s", backup_name)

    total_docs = 0
    total_questions = 0
    errors = []

    for idx, rfx_id in enumerate(rfx_ids, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Processing RFX %s", idx, len(rfx_ids), rfx_id)
        logger.info("=" * 60)

        # Step 1: Fetch raw data from API (single fetch, used for both ingest and ground truth)
        try:
            raw_data = fetch_live_rfx_data(rfx_id)
            rfx_name = raw_data.get("name", "Unknown RFX")
            logger.info("  Fetched: %s", rfx_name)
        except Exception as e:
            logger.error("  FAILED to fetch RFX %s: %s", rfx_id, e)
            errors.append(rfx_id)
            continue

        # Save raw snapshot for reproducibility
        snapshot_file = os.path.join(SNAPSHOT_DIR, f"rfx_{rfx_id}.json")
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False)

        # Step 2: Re-ingest into Cosmos DB
        if not skip_ingest:
            try:
                delete_rfx_documents(rfx_id)
                new_docs = parse_rfx_to_documents(raw_data)
                ingest_documents(new_docs)
                total_docs += len(new_docs)
                logger.info("  Ingested: %d documents", len(new_docs))
            except Exception as e:
                logger.error("  FAILED to ingest RFX %s: %s", rfx_id, e)
                errors.append(rfx_id)
                continue

        # Step 3: Generate ground truth from the SAME raw data
        try:
            rfx_summary = summarize_rfx_for_context(raw_data)

            # Extract supplier info for the ground truth file
            all_suppliers = raw_data.get("SupplierAnswers", [])
            supplier_names = [s.get("SupplierName", "Unknown") for s in all_suppliers]

            qa_data = generate_qa_pairs(rfx_summary, rfx_name, rfx_id)

            if qa_data and qa_data.get("questions"):
                # Build ground truth file
                gt_entry = {
                    "rfx_id": rfx_id,
                    "rfx_name": rfx_name,
                    "rfx_code": raw_data.get("code", ""),
                    "rfx_period": {
                        "start": (raw_data.get("starting") or "")[:10],
                        "end": (raw_data.get("ending") or "")[:10],
                    },
                    "suppliers_invited": supplier_names,
                    "total_questions_generated": len(qa_data["questions"]),
                    "questions": qa_data["questions"],
                }

                # Save ground truth
                safe_name = rfx_name.replace("/", "_").replace("\\", "_").replace(":", " -").replace("?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "_")
                gt_file = os.path.join(GROUND_TRUTH_DIR, f"rfx_{rfx_id}_{safe_name}.json")
                with open(gt_file, "w", encoding="utf-8") as f:
                    json.dump(gt_entry, f, indent=2, ensure_ascii=False)

                total_questions += len(qa_data["questions"])
                logger.info("  Ground truth: %d questions generated", len(qa_data["questions"]))
            else:
                logger.warning("  No Q&A pairs generated for RFX %s", rfx_id)

        except Exception as e:
            logger.error("  FAILED to generate ground truth for RFX %s: %s", rfx_id, e)
            errors.append(rfx_id)

        # Rate limiting
        time.sleep(1)

    logger.info("=" * 60)
    logger.info("REFRESH COMPLETE")
    logger.info("  RFXs processed: %d", len(rfx_ids) - len(errors))
    logger.info("  Documents ingested: %d", total_docs)
    logger.info("  Ground truth questions: %d", total_questions)
    if errors:
        logger.error("  Failed RFXs: %s", ", ".join(errors))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-ingest RFXs and regenerate ground truth")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Cosmos DB ingestion, only regenerate ground truth")
    parser.add_argument("--rfx-id", type=str, help="Process a single RFX only")
    args = parser.parse_args()

    refresh_all(skip_ingest=args.skip_ingest, rfx_id_filter=args.rfx_id)
