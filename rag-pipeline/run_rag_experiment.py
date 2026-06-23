"""
Run Ground Truth Questions Through the RAG Pipeline.

Sends all 344 ground truth questions through the baseline RAG pipeline,
records responses, response times, and metadata into per-RFX JSON files.

Usage:
    python run_rag_experiment.py                     # Run full experiment
    python run_rag_experiment.py --rfx-id 2100177    # Run single RFX
    python run_rag_experiment.py --experiment 2      # Custom experiment number
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langfuse import get_client, propagate_attributes
from src.retriever import query_rfx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress verbose Azure SDK HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truth_data")

langfuse = get_client()


def get_experiment_dir(experiment_num: int) -> str:
    """Get or create the experiment output directory."""
    dir_name = f"rag_responses_experiment_{experiment_num}"
    path = os.path.join(BASE_DIR, dir_name)
    os.makedirs(path, exist_ok=True)
    return path


def run_single_question(question: str, rfx_id: str, experiment_num: int = 1, question_id: int = 0) -> tuple[str, float, str]:
    """Run a single question through the RAG pipeline. Returns (response, elapsed_seconds, trace_id)."""
    # Deterministic trace ID so judge can attach scores to the same trace later
    seed = f"exp{experiment_num}_rfx{rfx_id}_q{question_id}"
    trace_id = langfuse.create_trace_id(seed=seed)

    start = time.time()
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="rag-experiment-query",
            trace_context={"trace_id": trace_id},
            input={"question": question, "rfx_id": rfx_id},
        ) as span:
            with propagate_attributes(
                session_id=f"experiment-{experiment_num}",
                metadata={"rfx_id": rfx_id, "experiment": str(experiment_num), "question_id": str(question_id)},
                tags=[f"experiment-{experiment_num}"],
            ):
                response = query_rfx(question, rfx_id)
            span.update(output={"response_length": len(response)})
    except Exception as e:
        logger.error("RAG query failed for RFX %s: %s", rfx_id, e)
        response = f"[ERROR: {e}]"
    elapsed = round(time.time() - start, 3)
    return response, elapsed, trace_id


def run_rfx(ground_truth_file: str, output_dir: str, experiment_num: int = 1) -> dict:
    """Run all questions for a single RFX through the RAG pipeline."""
    with open(ground_truth_file, "r", encoding="utf-8-sig") as f:
        gt_data = json.load(f)

    rfx_id = gt_data["rfx_id"]
    rfx_name = gt_data["rfx_name"]
    questions = gt_data.get("questions", [])

    if not questions:
        logger.warning("No questions in %s, skipping.", ground_truth_file)
        return None

    logger.info("Processing RFX: %s (%s) — %d questions", rfx_name, rfx_id, len(questions))

    responses = []
    total_time = 0.0

    for i, qa in enumerate(questions, 1):
        question = qa["question"]
        category = qa.get("category", "unknown")

        logger.info("  [%d/%d] (%s) %s", i, len(questions), category, question[:80])

        rag_response, elapsed, trace_id = run_single_question(question, rfx_id, experiment_num=experiment_num, question_id=qa.get("id", i))
        total_time += elapsed

        responses.append({
            "question_id": qa.get("id", i),
            "category": category,
            "question": question,
            "ground_truth_answer": qa["answer"],
            "rag_response": rag_response,
            "response_time_seconds": elapsed,
            "is_error": rag_response.startswith("[ERROR:"),
            "trace_id": trace_id,
        })

        logger.info("    → %.1fs | %d chars", elapsed, len(rag_response))

        # Brief pause to avoid rate limiting
        time.sleep(0.5)

    # Build output
    result = {
        "rfx_id": rfx_id,
        "rfx_name": rfx_name,
        "rfx_code": gt_data.get("rfx_code", ""),
        "experiment_timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "total_errors": sum(1 for r in responses if r["is_error"]),
        "total_response_time_seconds": round(total_time, 3),
        "average_response_time_seconds": round(total_time / len(questions), 3),
        "responses": responses,
    }

    # Save to file
    safe_name = rfx_name.replace("/", "_").replace("\\", "_")
    out_file = os.path.join(output_dir, f"rfx_{rfx_id}_{safe_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("  Saved: %s (avg %.1fs/question)", os.path.basename(out_file), result["average_response_time_seconds"])
    return result


def run_experiment(experiment_num: int = 1, rfx_id_filter: str = None):
    """Run the full experiment across all ground truth files."""
    output_dir = get_experiment_dir(experiment_num)
    logger.info("Experiment %d — Output: %s", experiment_num, output_dir)

    # Collect ground truth files
    gt_files = sorted([
        os.path.join(GROUND_TRUTH_DIR, f)
        for f in os.listdir(GROUND_TRUTH_DIR)
        if f.startswith("rfx_") and f.endswith(".json") and not f.startswith("_")
    ])

    if rfx_id_filter:
        gt_files = [f for f in gt_files if rfx_id_filter in os.path.basename(f)]
        if not gt_files:
            logger.error("No ground truth file found for RFX %s", rfx_id_filter)
            return

    logger.info("Found %d ground truth files to process", len(gt_files))

    # Check for already-completed files (resume support)
    existing = set()
    for f in os.listdir(output_dir):
        if f.startswith("rfx_") and f.endswith(".json"):
            existing.add(f)

    all_results = []
    experiment_start = time.time()

    for idx, gt_file in enumerate(gt_files, 1):
        # Check if output already exists
        base = os.path.basename(gt_file)
        if base in existing:
            logger.info("[%d/%d] SKIPPING (already done): %s", idx, len(gt_files), base)
            # Load existing result for summary
            out_path = os.path.join(output_dir, base)
            with open(out_path, "r", encoding="utf-8-sig") as f:
                all_results.append(json.load(f))
            continue

        logger.info("=" * 60)
        logger.info("[%d/%d] %s", idx, len(gt_files), os.path.basename(gt_file))
        logger.info("=" * 60)

        result = run_rfx(gt_file, output_dir, experiment_num=experiment_num)
        if result:
            all_results.append(result)

    experiment_elapsed = round(time.time() - experiment_start, 1)

    # Write experiment summary
    total_q = sum(r["total_questions"] for r in all_results)
    total_err = sum(r["total_errors"] for r in all_results)
    total_time = sum(r["total_response_time_seconds"] for r in all_results)
    avg_time = round(total_time / total_q, 3) if total_q > 0 else 0

    summary = {
        "experiment": experiment_num,
        "experiment_timestamp": datetime.now().isoformat(),
        "experiment_wall_time_seconds": experiment_elapsed,
        "total_rfx_processed": len(all_results),
        "total_questions": total_q,
        "total_errors": total_err,
        "total_rag_time_seconds": round(total_time, 1),
        "average_response_time_seconds": avg_time,
        "per_rfx_summary": [
            {
                "rfx_id": r["rfx_id"],
                "rfx_name": r["rfx_name"],
                "questions": r["total_questions"],
                "errors": r["total_errors"],
                "avg_response_time": r["average_response_time_seconds"],
            }
            for r in all_results
        ],
    }

    summary_file = os.path.join(output_dir, "_experiment_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("EXPERIMENT %d COMPLETE", experiment_num)
    logger.info("  RFXs: %d | Questions: %d | Errors: %d", len(all_results), total_q, total_err)
    logger.info("  Total RAG time: %.1fs | Avg: %.1fs/question", total_time, avg_time)
    logger.info("  Wall time: %.1fs", experiment_elapsed)
    logger.info("  Summary: %s", summary_file)
    logger.info("=" * 60)

    # Flush all Langfuse traces
    langfuse.flush()
    logger.info("Langfuse traces flushed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ground truth through RAG pipeline")
    parser.add_argument("--rfx-id", type=str, help="Run only for a specific RFX ID")
    parser.add_argument("--experiment", type=int, default=1, help="Experiment number (default: 1)")
    args = parser.parse_args()

    run_experiment(experiment_num=args.experiment, rfx_id_filter=args.rfx_id)
