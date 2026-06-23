"""
LLM-as-a-Judge Experiment Evaluator.

Reads RAG pipeline responses from an experiment folder, compares each response
against the ground truth answer using an LLM judge, and saves consolidated
results into the experiment_results folder.

Usage:
    python run_judge_evaluation.py                     # Evaluate experiment 1
    python run_judge_evaluation.py --experiment 2      # Evaluate experiment 2
    python run_judge_evaluation.py --rfx-id 2100177    # Evaluate single RFX
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langfuse import get_client
from src.config import (
    AI_FOUNDRY_CHAT_ENDPOINT,
    AI_FOUNDRY_CHAT_KEY,
    AI_FOUNDRY_CHAT_MODEL,
)
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

langfuse = get_client()

# Initialize judge LLM client
judge_client = ChatCompletionsClient(
    endpoint=AI_FOUNDRY_CHAT_ENDPOINT,
    credential=AzureKeyCredential(AI_FOUNDRY_CHAT_KEY),
    api_version="2025-01-01-preview",
)

# ============================================================
# Evaluation Criteria & Judge Prompt
# ============================================================
EVALUATION_CRITERIA = [
    "precision", "accuracy", "faithfulness", "completeness", "relevance", "coherence"
]

JUDGE_SYSTEM_PROMPT = """You are an impartial, expert evaluator (LLM-as-a-Judge) for a Procurement RAG system.

Your task is to evaluate a RAG system's response against a ground truth answer for a given question about procurement RFX data.

EVALUATION CRITERIA (score each 1-10):

1. PRECISION (1-10): Does the response answer EXACTLY what was asked without including irrelevant details?
   - 10: Perfectly targeted, no extraneous information
   - 7: Mostly precise, minor irrelevant details
   - 4: Partially relevant, significant off-topic content
   - 1: Completely off-topic or vague

2. ACCURACY (1-10): Are all facts, numbers, and claims factually correct?
   - 10: All facts verified correct against ground truth
   - 7: Minor factual errors that don't change the conclusion
   - 4: Multiple factual errors or one major error
   - 1: Mostly incorrect or fabricated information

3. FAITHFULNESS (1-10): Is the response faithful to source data with no hallucination?
   - 10: Every claim is traceable to source data
   - 7: Mostly grounded, minor unsupported inferences
   - 4: Contains hallucinated facts mixed with real data
   - 1: Predominantly hallucinated/invented content

4. COMPLETENESS (1-10): Does the response cover all key points from the ground truth?
   - 10: All important details from ground truth are present
   - 7: Most key points covered, minor omissions
   - 4: Significant information missing
   - 1: Nearly empty or misses the entire point

5. RELEVANCE (1-10): How directly does the response address the question?
   - 10: Directly and fully addresses the question
   - 7: Addresses the question but includes tangential info
   - 4: Partially addresses the question
   - 1: Does not address the question at all

6. COHERENCE (1-10): Is the response well-structured and easy to understand?
   - 10: Excellent structure, logical flow, clear language
   - 7: Good structure with minor organization issues
   - 4: Disorganized or hard to follow
   - 1: Incoherent or garbled text

OUTPUT FORMAT (strict JSON only, no other text):
{
  "scores": {
    "precision": <1-10>,
    "accuracy": <1-10>,
    "faithfulness": <1-10>,
    "completeness": <1-10>,
    "relevance": <1-10>,
    "coherence": <1-10>
  },
  "overall_score": <1-10 weighted average>,
  "reasoning": "Brief explanation of the scores (2-3 sentences)",
  "issues": ["List of specific issues found, if any"]
}

IMPORTANT:
- Be STRICT but fair. Only give 9-10 for truly excellent responses.
- Penalize hallucination heavily (faithfulness).
- A response saying "I cannot find this information" when data IS available should score low on completeness.
- A response saying "I cannot find this information" when data truly ISN'T available should score HIGH on faithfulness.
- Consider the procurement domain context when evaluating."""


def judge_response(question: str, ground_truth: str, rag_response: str, evidence: str = "") -> dict:
    """Use the Judge LLM to evaluate a single RAG response against ground truth."""
    user_prompt = f"""Evaluate the following RAG system response:

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

SUPPORTING EVIDENCE: {evidence}

RAG SYSTEM RESPONSE: {rag_response}

Provide your evaluation as JSON only."""

    try:
        response = judge_client.complete(
            messages=[
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                UserMessage(content=user_prompt),
            ],
            model=AI_FOUNDRY_CHAT_MODEL,
            temperature=0.1,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse judge response: %s", e)
        return {
            "scores": {c: 0 for c in EVALUATION_CRITERIA},
            "overall_score": 0,
            "reasoning": f"Judge response parsing failed: {e}",
            "issues": ["evaluation_error"],
        }
    except Exception as e:
        logger.error("Judge evaluation failed: %s", e)
        return {
            "scores": {c: 0 for c in EVALUATION_CRITERIA},
            "overall_score": 0,
            "reasoning": f"Judge evaluation error: {e}",
            "issues": ["evaluation_error"],
        }


def run_evaluation(experiment_num: int = 1, rfx_id_filter: str = None):
    """Run LLM judge evaluation on all RAG responses from an experiment."""
    responses_dir = os.path.join(BASE_DIR, f"rag_responses_experiment_{experiment_num}")
    results_dir = os.path.join(BASE_DIR, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(responses_dir):
        logger.error("Experiment folder not found: %s", responses_dir)
        return

    # Collect response files
    response_files = sorted([
        os.path.join(responses_dir, f)
        for f in os.listdir(responses_dir)
        if f.startswith("rfx_") and f.endswith(".json")
    ])

    if rfx_id_filter:
        response_files = [f for f in response_files if rfx_id_filter in os.path.basename(f)]
        if not response_files:
            logger.error("No response file found for RFX %s", rfx_id_filter)
            return

    logger.info("=" * 60)
    logger.info("LLM-AS-A-JUDGE EVALUATION — Experiment %d", experiment_num)
    logger.info("Response files: %d", len(response_files))
    logger.info("=" * 60)

    all_evaluations = []
    per_rfx_results = []
    eval_start = time.time()

    for file_idx, resp_file in enumerate(response_files, 1):
        with open(resp_file, "r", encoding="utf-8-sig") as f:
            resp_data = json.load(f)

        rfx_id = resp_data["rfx_id"]
        rfx_name = resp_data["rfx_name"]
        responses = resp_data.get("responses", [])

        if not responses:
            logger.warning("No responses in %s, skipping.", resp_file)
            continue

        logger.info("[%d/%d] Judging RFX: %s (%s) — %d responses",
                     file_idx, len(response_files), rfx_name, rfx_id, len(responses))

        rfx_evaluations = []

        for i, resp in enumerate(responses, 1):
            question = resp["question"]
            ground_truth = resp["ground_truth_answer"]
            rag_response = resp["rag_response"]
            category = resp.get("category", "unknown")
            response_time = resp.get("response_time_seconds", 0)

            logger.info("  [%d/%d] (%s) %s", i, len(responses), category, question[:70])

            # Skip error responses from judging
            if resp.get("is_error", False):
                judgment = {
                    "scores": {c: 0 for c in EVALUATION_CRITERIA},
                    "overall_score": 0,
                    "reasoning": "RAG pipeline returned an error, cannot evaluate.",
                    "issues": ["rag_error"],
                }
            else:
                judgment = judge_response(question, ground_truth, rag_response)
                time.sleep(0.5)  # Rate limiting

            evaluation = {
                "rfx_id": rfx_id,
                "rfx_name": rfx_name,
                "question_id": resp.get("question_id", i),
                "category": category,
                "question": question,
                "ground_truth_answer": ground_truth,
                "rag_response": rag_response,
                "rag_response_time_seconds": response_time,
                "judgment": judgment,
            }

            rfx_evaluations.append(evaluation)
            all_evaluations.append(evaluation)

            score = judgment.get("overall_score", 0)
            logger.info("    → Score: %.1f/10", score)

            # Push scores to Langfuse trace (if trace_id was captured during RAG experiment)
            trace_id = resp.get("trace_id")
            if trace_id and score > 0:
                for criterion, criterion_score in judgment.get("scores", {}).items():
                    langfuse.create_score(
                        trace_id=trace_id,
                        name=criterion,
                        value=float(criterion_score),
                        data_type="NUMERIC",
                        comment=judgment.get("reasoning", "")[:200],
                        metadata={"rfx_id": rfx_id, "category": category, "experiment": str(experiment_num)},
                    )
                langfuse.create_score(
                    trace_id=trace_id,
                    name="overall_score",
                    value=float(score),
                    data_type="NUMERIC",
                    comment=judgment.get("reasoning", "")[:200],
                    metadata={"rfx_id": rfx_id, "category": category, "experiment": str(experiment_num)},
                )

        # Per-RFX aggregate
        valid = [e for e in rfx_evaluations if e["judgment"]["overall_score"] > 0]
        if valid:
            avg_scores = {}
            for criterion in EVALUATION_CRITERIA:
                vals = [e["judgment"]["scores"].get(criterion, 0) for e in valid]
                avg_scores[criterion] = round(sum(vals) / len(vals), 2)
            avg_overall = round(sum(e["judgment"]["overall_score"] for e in valid) / len(valid), 2)
        else:
            avg_scores = {c: 0 for c in EVALUATION_CRITERIA}
            avg_overall = 0

        # Category breakdown for this RFX
        cat_scores = {}
        for e in valid:
            cat = e["category"]
            if cat not in cat_scores:
                cat_scores[cat] = []
            cat_scores[cat].append(e["judgment"]["overall_score"])

        per_rfx_results.append({
            "rfx_id": rfx_id,
            "rfx_name": rfx_name,
            "total_questions": len(responses),
            "evaluated": len(valid),
            "errors": len(rfx_evaluations) - len(valid),
            "average_scores": avg_scores,
            "average_overall_score": avg_overall,
            "category_averages": {
                cat: round(sum(s) / len(s), 2)
                for cat, s in cat_scores.items()
            },
        })

        logger.info("  → RFX %s avg score: %.1f/10", rfx_id, avg_overall)

    eval_elapsed = round(time.time() - eval_start, 1)

    # ── Global aggregates ──
    all_valid = [e for e in all_evaluations if e["judgment"]["overall_score"] > 0]

    global_scores = {}
    for criterion in EVALUATION_CRITERIA:
        vals = [e["judgment"]["scores"].get(criterion, 0) for e in all_valid]
        global_scores[criterion] = round(sum(vals) / len(vals), 2) if vals else 0

    global_overall = round(
        sum(e["judgment"]["overall_score"] for e in all_valid) / len(all_valid), 2
    ) if all_valid else 0

    # Global category breakdown
    global_cat_scores = {}
    for e in all_valid:
        cat = e["category"]
        if cat not in global_cat_scores:
            global_cat_scores[cat] = []
        global_cat_scores[cat].append(e["judgment"]["overall_score"])

    global_category_averages = {
        cat: round(sum(s) / len(s), 2)
        for cat, s in global_cat_scores.items()
    }

    # ── Build final result file ──
    result = {
        "experiment": experiment_num,
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_wall_time_seconds": eval_elapsed,
        "judge_model": AI_FOUNDRY_CHAT_MODEL,
        "judge_temperature": 0.1,
        "total_questions": len(all_evaluations),
        "total_evaluated": len(all_valid),
        "total_errors": len(all_evaluations) - len(all_valid),
        "global_average_scores": global_scores,
        "global_overall_score": global_overall,
        "global_category_averages": global_category_averages,
        "per_rfx_summary": per_rfx_results,
        "evaluations": all_evaluations,
    }

    # Save
    result_file = os.path.join(
        results_dir,
        f"experiment_{experiment_num}_results.json"
    )
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE — Experiment %d", experiment_num)
    logger.info("  Questions: %d | Evaluated: %d | Errors: %d",
                len(all_evaluations), len(all_valid), len(all_evaluations) - len(all_valid))
    logger.info("  Global Overall Score: %.2f / 10", global_overall)
    logger.info("  Criteria Scores:")
    for c, s in global_scores.items():
        logger.info("    %-15s %.2f", c, s)
    logger.info("  Category Performance:")
    for cat, s in sorted(global_category_averages.items()):
        logger.info("    %-20s %.2f", cat, s)
    logger.info("  Wall time: %.1fs", eval_elapsed)
    logger.info("  Results: %s", result_file)
    logger.info("=" * 60)

    # Flush all Langfuse scores
    langfuse.flush()
    logger.info("Langfuse scores flushed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Experiment Evaluator")
    parser.add_argument("--experiment", type=int, default=1, help="Experiment number to evaluate")
    parser.add_argument("--rfx-id", type=str, help="Evaluate only a specific RFX ID")
    args = parser.parse_args()

    run_evaluation(experiment_num=args.experiment, rfx_id_filter=args.rfx_id)
