"""
Langfuse Dataset & Experiment Management.

1. Upload ground truth Q&A pairs as a Langfuse Dataset
2. Backfill existing experiment results as a Langfuse Dataset Run
3. Run new experiments using Langfuse's experiment framework

Usage:
    # Upload ground truth as a Langfuse dataset
    python langfuse_experiment.py upload-dataset

    # Backfill experiment 1 results as a dataset run (no re-run needed)
    python langfuse_experiment.py backfill --experiment 1

    # Run a new experiment through the Langfuse experiment framework
    python langfuse_experiment.py run --experiment 2

    # Run with a specific RFX only
    python langfuse_experiment.py run --experiment 2 --rfx-id 2100177
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config  # noqa: F401 — loads local.settings.json env vars
from langfuse import get_client, propagate_attributes
from langfuse.experiment import Evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truth_data")

langfuse = get_client()

DATASET_NAME = "procurement-rfx-qa-v2"
EVALUATION_CRITERIA = ["precision", "accuracy", "faithfulness", "completeness", "relevance", "coherence"]


# ============================================================
# 1. Upload Ground Truth as Langfuse Dataset
# ============================================================

def upload_dataset():
    """Upload all ground truth Q&A pairs as a Langfuse Dataset."""
    logger.info("Creating Langfuse dataset: %s", DATASET_NAME)

    # Count actual items from ground truth files
    gt_files = sorted([
        os.path.join(GROUND_TRUTH_DIR, f)
        for f in os.listdir(GROUND_TRUTH_DIR)
        if f.startswith("rfx_") and f.endswith(".json")
    ])
    total_q = 0
    categories = set()
    for gf in gt_files:
        with open(gf, "r", encoding="utf-8-sig") as fh:
            d = json.load(fh)
        for qa in d.get("questions", []):
            total_q += 1
            categories.add(qa.get("category", "unknown"))

    langfuse.create_dataset(
        name=DATASET_NAME,
        description=f"{total_q} procurement RFX Q&A pairs across {len(gt_files)} RFXs",
        metadata={
            "total_rfx": len(gt_files),
            "total_questions": total_q,
            "categories": sorted(categories),
        },
    )

    gt_files = sorted([
        os.path.join(GROUND_TRUTH_DIR, f)
        for f in os.listdir(GROUND_TRUTH_DIR)
        if f.startswith("rfx_") and f.endswith(".json")
    ])

    total_items = 0
    for gt_file in gt_files:
        with open(gt_file, "r", encoding="utf-8-sig") as f:
            gt_data = json.load(f)

        rfx_id = gt_data["rfx_id"]
        rfx_name = gt_data["rfx_name"]
        questions = gt_data.get("questions", [])

        logger.info("  RFX %s (%s): %d questions", rfx_id, rfx_name, len(questions))

        for qa in questions:
            question_id = qa.get("id", 0)
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                id=f"v2_rfx{rfx_id}_q{question_id}",
                input={
                    "question": qa["question"],
                    "rfx_id": rfx_id,
                },
                expected_output={
                    "answer": qa["answer"],
                    "evidence": qa.get("evidence", ""),
                },
                metadata={
                    "rfx_id": rfx_id,
                    "rfx_name": rfx_name,
                    "question_id": question_id,
                    "category": qa.get("category", "unknown"),
                },
            )
            total_items += 1

    langfuse.flush()
    logger.info("=" * 60)
    logger.info("DATASET UPLOAD COMPLETE")
    logger.info("  Dataset: %s", DATASET_NAME)
    logger.info("  Items: %d", total_items)
    logger.info("  View at: https://cloud.langfuse.com → Datasets")
    logger.info("=" * 60)


# ============================================================
# 2. Backfill Existing Experiment as a Dataset Run
# ============================================================

def backfill_experiment(experiment_num: int = 1):
    """Backfill existing experiment results as a Langfuse dataset run.
    
    Uses dataset.run_experiment() with a task that returns pre-existing results
    and evaluators that return pre-existing scores. No re-running needed.
    """
    results_file = os.path.join(BASE_DIR, "experiment_results", f"experiment_{experiment_num}_results.json")
    if not os.path.exists(results_file):
        logger.error("Results file not found: %s", results_file)
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluations = results.get("evaluations", [])
    run_name = f"experiment-{experiment_num}-baseline"

    logger.info("=" * 60)
    logger.info("BACKFILLING EXPERIMENT %d AS DATASET RUN: %s", experiment_num, run_name)
    logger.info("Evaluations: %d", len(evaluations))
    logger.info("=" * 60)

    # Get the dataset
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        logger.error("Dataset '%s' not found. Run 'upload-dataset' first. Error: %s", DATASET_NAME, e)
        return

    # Build lookup from (rfx_id, question_id) → evaluation result
    eval_lookup = {}
    for ev in evaluations:
        key = (ev["rfx_id"], ev.get("question_id", 0))
        eval_lookup[key] = ev

    # Task: return pre-existing RAG response (no pipeline call)
    def backfill_task(*, item, **kwargs):
        rfx_id = item.input.get("rfx_id", "")
        q_id = item.metadata.get("question_id", 0)
        ev = eval_lookup.get((rfx_id, q_id))
        if ev:
            return {
                "answer": ev.get("rag_response", ""),
                "response_time_seconds": ev.get("rag_response_time_seconds", 0),
            }
        return {"answer": "[NOT FOUND IN EXPERIMENT RESULTS]", "response_time_seconds": 0}

    # Evaluator: return pre-existing judge scores (no LLM call)
    def backfill_evaluator(*, output, expected_output, input, metadata, **kwargs):
        rfx_id = input.get("rfx_id", "")
        q_id = metadata.get("question_id", 0)
        ev = eval_lookup.get((rfx_id, q_id))
        if not ev:
            return []

        judgment = ev.get("judgment", {})
        overall_score = judgment.get("overall_score", 0)
        if overall_score == 0:
            return []

        evals = []
        for criterion in EVALUATION_CRITERIA:
            score_val = judgment.get("scores", {}).get(criterion, 0)
            evals.append(Evaluation(
                name=criterion,
                value=float(score_val),
                data_type="NUMERIC",
                comment=judgment.get("reasoning", "")[:200],
                metadata={"category": metadata.get("category", ""), "experiment": str(experiment_num)},
            ))

        evals.append(Evaluation(
            name="overall_score",
            value=float(overall_score),
            data_type="NUMERIC",
            comment=judgment.get("reasoning", "")[:200],
            metadata={"category": metadata.get("category", ""), "experiment": str(experiment_num)},
        ))
        return evals

    # Run-level evaluator: compute averages
    def compute_averages(*, item_results, **kwargs):
        all_evals = []
        for result in item_results:
            all_evals.extend(result.evaluations)
        evals = []
        for criterion in EVALUATION_CRITERIA + ["overall_score"]:
            vals = [e.value for e in all_evals if e.name == criterion and e.value > 0]
            if vals:
                evals.append(Evaluation(
                    name=f"avg_{criterion}",
                    value=round(sum(vals) / len(vals), 2),
                    data_type="NUMERIC",
                    comment=f"Average {criterion} across {len(vals)} items",
                ))
        return evals

    result = dataset.run_experiment(
        name=run_name,
        description=f"Baseline experiment {experiment_num} (backfilled from local results)",
        task=backfill_task,
        evaluators=[backfill_evaluator],
        run_evaluators=[compute_averages],
        max_concurrency=10,
        metadata={"experiment": str(experiment_num), "backfilled": "true"},
    )

    langfuse.flush()

    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(result.format())
    if hasattr(result, "dataset_run_url") and result.dataset_run_url:
        logger.info("  View results: %s", result.dataset_run_url)
    logger.info("  Compare runs: https://cloud.langfuse.com → Datasets → %s", DATASET_NAME)
    logger.info("=" * 60)


# ============================================================
# 3. Run New Experiment via Langfuse Experiment Framework
# ============================================================

def run_new_experiment(experiment_num: int = 2, rfx_id_filter: str = None):
    """Run a new experiment using Langfuse's dataset.run_experiment() framework.
    
    This sends all dataset items through the RAG pipeline and evaluates them
    using the LLM-as-a-Judge, all tracked natively in Langfuse with side-by-side
    comparison against previous runs.
    """
    from src.retriever import query_rfx
    from run_judge_evaluation import judge_response

    # Get dataset
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        logger.error("Dataset '%s' not found. Run 'upload-dataset' first. Error: %s", DATASET_NAME, e)
        return

    # Filter items if rfx_id specified
    if rfx_id_filter:
        original_items = dataset.items
        dataset.items = [item for item in original_items if item.input.get("rfx_id") == rfx_id_filter]
        logger.info("Filtered to %d items for RFX %s", len(dataset.items), rfx_id_filter)

    logger.info("=" * 60)
    logger.info("RUNNING EXPERIMENT %d VIA LANGFUSE FRAMEWORK", experiment_num)
    logger.info("Dataset: %s (%d items)", DATASET_NAME, len(dataset.items))
    logger.info("=" * 60)

    # --- Task function: runs the RAG pipeline ---
    def rag_task(*, item, **kwargs):
        question = item.input["question"]
        rfx_id = item.input["rfx_id"]

        with propagate_attributes(
            session_id=f"experiment-{experiment_num}",
            metadata={
                "rfx_id": rfx_id,
                "experiment": str(experiment_num),
                "question_id": str(item.metadata.get("question_id", "")),
                "category": item.metadata.get("category", ""),
            },
            tags=[f"experiment-{experiment_num}", f"rfx-{rfx_id}", item.metadata.get("category", "")],
        ):
            start = time.time()
            response = query_rfx(question, rfx_id)
            elapsed = round(time.time() - start, 3)

        return {
            "answer": response,
            "response_time_seconds": elapsed,
        }

    # --- Evaluator functions: score using LLM-as-a-Judge ---
    def judge_evaluator(*, output, expected_output, input, metadata, **kwargs):
        """Run LLM-as-a-Judge and return all scores as separate Evaluations."""
        question = input["question"]
        ground_truth = expected_output["answer"]
        evidence = expected_output.get("evidence", "")
        rag_response = output["answer"]

        judgment = judge_response(question, ground_truth, rag_response, evidence)
        time.sleep(0.3)  # Rate limiting

        evaluations = []
        for criterion in EVALUATION_CRITERIA:
            score_val = judgment.get("scores", {}).get(criterion, 0)
            evaluations.append(Evaluation(
                name=criterion,
                value=float(score_val),
                data_type="NUMERIC",
                comment=judgment.get("reasoning", "")[:200],
                metadata={"category": metadata.get("category", "")},
            ))

        evaluations.append(Evaluation(
            name="overall_score",
            value=float(judgment.get("overall_score", 0)),
            data_type="NUMERIC",
            comment=judgment.get("reasoning", "")[:200],
            metadata={"category": metadata.get("category", "")},
        ))

        return evaluations

    # --- Run-level evaluators: aggregate statistics ---
    def compute_averages(*, item_results, **kwargs):
        """Compute average scores across all items for the entire run."""
        all_evals = []
        for result in item_results:
            all_evals.extend(result.evaluations)
        evaluations = []
        for criterion in EVALUATION_CRITERIA + ["overall_score"]:
            vals = [e.value for e in all_evals if e.name == criterion and e.value > 0]
            if vals:
                evaluations.append(Evaluation(
                    name=f"avg_{criterion}",
                    value=round(sum(vals) / len(vals), 2),
                    data_type="NUMERIC",
                    comment=f"Average {criterion} across {len(vals)} items",
                ))
        return evaluations

    # --- Execute ---
    run_name = f"experiment-{experiment_num}"
    result = dataset.run_experiment(
        name=run_name,
        description=f"Experiment {experiment_num} — RAG pipeline evaluation with LLM-as-a-Judge",
        task=rag_task,
        evaluators=[judge_evaluator],
        run_evaluators=[compute_averages],
        max_concurrency=3,  # Conservative to respect Azure rate limits
        metadata={"experiment": str(experiment_num)},
    )

    langfuse.flush()

    logger.info("=" * 60)
    logger.info("EXPERIMENT %d COMPLETE", experiment_num)
    logger.info(result.format())
    if hasattr(result, "dataset_run_url") and result.dataset_run_url:
        logger.info("  View results: %s", result.dataset_run_url)
    logger.info("  Compare runs: https://cloud.langfuse.com → Datasets → %s", DATASET_NAME)
    logger.info("=" * 60)

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Langfuse Dataset & Experiment Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # upload-dataset
    subparsers.add_parser("upload-dataset", help="Upload ground truth as a Langfuse Dataset")

    # backfill
    bf = subparsers.add_parser("backfill", help="Backfill existing experiment as a Langfuse Dataset Run")
    bf.add_argument("--experiment", type=int, default=1, help="Experiment number to backfill")

    # run
    run = subparsers.add_parser("run", help="Run a new experiment via Langfuse framework")
    run.add_argument("--experiment", type=int, default=2, help="Experiment number")
    run.add_argument("--rfx-id", type=str, help="Run only a specific RFX")

    args = parser.parse_args()

    if args.command == "upload-dataset":
        upload_dataset()
    elif args.command == "backfill":
        backfill_experiment(experiment_num=args.experiment)
    elif args.command == "run":
        run_new_experiment(experiment_num=args.experiment, rfx_id_filter=args.rfx_id)
