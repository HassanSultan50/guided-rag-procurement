"""
Upload existing Experiment 1 results to Langfuse.

Reads RAG responses and judge scores from local JSON files and creates
Langfuse traces with attached evaluation scores — no need to re-run the experiment.

Usage:
    python upload_experiment_to_langfuse.py
    python upload_experiment_to_langfuse.py --experiment 1
    python upload_experiment_to_langfuse.py --rfx-id 2100177
"""

import json
import os
import sys
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first to load local.settings.json env vars (including Langfuse keys)
import src.config  # noqa: F401
from langfuse import get_client, propagate_attributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
langfuse = get_client()

EVALUATION_CRITERIA = ["precision", "accuracy", "faithfulness", "completeness", "relevance", "coherence"]


def upload_experiment(experiment_num: int = 1, rfx_id_filter: str = None):
    """Upload existing experiment results to Langfuse as traces with scores."""

    # Load evaluation results
    results_file = os.path.join(BASE_DIR, "experiment_results", f"experiment_{experiment_num}_results.json")
    if not os.path.exists(results_file):
        logger.error("Results file not found: %s", results_file)
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluations = results.get("evaluations", [])
    if rfx_id_filter:
        evaluations = [e for e in evaluations if e["rfx_id"] == rfx_id_filter]

    logger.info("=" * 60)
    logger.info("UPLOADING EXPERIMENT %d TO LANGFUSE", experiment_num)
    logger.info("Evaluations to upload: %d", len(evaluations))
    logger.info("=" * 60)

    # Load RAG responses for additional context (response text)
    rag_responses_dir = os.path.join(BASE_DIR, f"rag_responses_experiment_{experiment_num}")
    rag_data = {}
    if os.path.isdir(rag_responses_dir):
        for f_name in os.listdir(rag_responses_dir):
            if f_name.startswith("rfx_") and f_name.endswith(".json"):
                fpath = os.path.join(rag_responses_dir, f_name)
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                rfx_id = data["rfx_id"]
                for resp in data.get("responses", []):
                    key = (rfx_id, resp.get("question_id", 0))
                    rag_data[key] = resp

    uploaded = 0
    errors = 0

    for i, evaluation in enumerate(evaluations, 1):
        rfx_id = evaluation["rfx_id"]
        rfx_name = evaluation.get("rfx_name", "")
        question_id = evaluation.get("question_id", i)
        question = evaluation["question"]
        category = evaluation.get("category", "unknown")
        judgment = evaluation.get("judgment", {})
        overall_score = judgment.get("overall_score", 0)

        # Get RAG response from evaluation or rag_data
        rag_response = evaluation.get("rag_response", "")
        if not rag_response:
            resp_data = rag_data.get((rfx_id, question_id), {})
            rag_response = resp_data.get("rag_response", "")

        ground_truth = evaluation.get("ground_truth_answer", "")
        response_time = evaluation.get("rag_response_time_seconds", 0)

        # Deterministic trace ID matching run_rag_experiment.py format
        seed = f"exp{experiment_num}_rfx{rfx_id}_q{question_id}"
        trace_id = langfuse.create_trace_id(seed=seed)

        try:
            # Create a trace representing this RAG query
            with langfuse.start_as_current_observation(
                as_type="span",
                name="query_rfx",
                trace_context={"trace_id": trace_id},
                input={"question": question, "rfx_id": rfx_id, "ground_truth": ground_truth},
                output={"answer": rag_response, "response_time_seconds": response_time},
            ) as span:
                with propagate_attributes(
                    session_id=f"experiment-{experiment_num}",
                    metadata={
                        "rfx_id": rfx_id,
                        "rfx_name": rfx_name,
                        "experiment": str(experiment_num),
                        "question_id": str(question_id),
                        "category": category,
                    },
                    tags=[f"experiment-{experiment_num}", f"rfx-{rfx_id}", category],
                ):
                    pass  # Trace is created with input/output already

            # Push evaluation scores to the trace
            if overall_score > 0:
                for criterion in EVALUATION_CRITERIA:
                    criterion_score = judgment.get("scores", {}).get(criterion, 0)
                    if criterion_score > 0:
                        langfuse.create_score(
                            trace_id=trace_id,
                            name=criterion,
                            value=float(criterion_score),
                            data_type="NUMERIC",
                            comment=judgment.get("reasoning", "")[:200],
                            metadata={
                                "rfx_id": rfx_id,
                                "category": category,
                                "experiment": str(experiment_num),
                            },
                        )

                langfuse.create_score(
                    trace_id=trace_id,
                    name="overall_score",
                    value=float(overall_score),
                    data_type="NUMERIC",
                    comment=judgment.get("reasoning", "")[:200],
                    metadata={
                        "rfx_id": rfx_id,
                        "category": category,
                        "experiment": str(experiment_num),
                    },
                )

            uploaded += 1

            if i % 25 == 0:
                logger.info("  [%d/%d] Uploaded... (flushing batch)", i, len(evaluations))
                langfuse.flush()

        except Exception as e:
            logger.error("  Failed to upload eval %d (RFX %s, Q%d): %s", i, rfx_id, question_id, e)
            errors += 1

    # Final flush
    langfuse.flush()

    logger.info("=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info("  Uploaded: %d traces with scores", uploaded)
    logger.info("  Errors: %d", errors)
    logger.info("  View at: https://cloud.langfuse.com")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload experiment results to Langfuse")
    parser.add_argument("--experiment", type=int, default=1, help="Experiment number")
    parser.add_argument("--rfx-id", type=str, help="Upload only a specific RFX ID")
    args = parser.parse_args()

    upload_experiment(experiment_num=args.experiment, rfx_id_filter=args.rfx_id)
