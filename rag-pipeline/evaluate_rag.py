"""
LLM-as-a-Judge Evaluation Framework.

Evaluates RAG system responses against ground truth Q&A pairs using an LLM judge.
The judge scores responses on a 1-10 scale across multiple quality criteria.

Usage:
    python evaluate_rag.py                    # Run full evaluation
    python evaluate_rag.py --rfx-id 123       # Evaluate single RFX
    python evaluate_rag.py --report           # Generate report from existing results
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
from src.retriever import query_rfx
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

langfuse = get_client()

# Initialize judge LLM client
judge_client = ChatCompletionsClient(
    endpoint=AI_FOUNDRY_CHAT_ENDPOINT,
    credential=AzureKeyCredential(AI_FOUNDRY_CHAT_KEY),
    api_version="2025-01-01-preview",
)

# ============================================================
# Evaluation Criteria
# ============================================================
EVALUATION_CRITERIA = {
    "precision": "How precisely does the response answer the specific question asked? Does it avoid including irrelevant information?",
    "accuracy": "Are the facts, numbers, dates, and claims in the response factually correct compared to the ground truth?",
    "faithfulness": "Is the response faithful to the source data? Does it avoid hallucinating or making up information not in the data?",
    "completeness": "Does the response cover all key points present in the ground truth answer? Are important details missing?",
    "relevance": "How relevant is the response to the question? Does it directly address what was asked?",
    "coherence": "Is the response well-structured, logically organized, and easy to understand?",
}

# ============================================================
# Judge System Prompt
# ============================================================
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


# ============================================================
# Judge Evaluation
# ============================================================
def evaluate_single_response(question: str, ground_truth: str, rag_response: str, evidence: str = "") -> dict:
    """
    Uses the Judge LLM to evaluate a RAG response against ground truth.
    Returns scores and reasoning.
    """
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

        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse judge response: %s", e)
        return {
            "scores": {k: 0 for k in EVALUATION_CRITERIA},
            "overall_score": 0,
            "reasoning": f"Judge response parsing failed: {e}",
            "issues": ["evaluation_error"],
        }
    except Exception as e:
        logger.error("Judge evaluation failed: %s", e)
        return {
            "scores": {k: 0 for k in EVALUATION_CRITERIA},
            "overall_score": 0,
            "reasoning": f"Judge evaluation error: {e}",
            "issues": ["evaluation_error"],
        }


# ============================================================
# RAG System Querying
# ============================================================
def get_rag_response(question: str, rfx_id: str) -> tuple[str, str | None]:
    """Gets the RAG system's response to a question for a given RFX. Returns (response, trace_id)."""
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="eval-rag-query",
            input={"question": question, "rfx_id": rfx_id},
        ) as span:
            response = query_rfx(question, rfx_id)
            trace_id = langfuse.get_current_trace_id()
            span.update(output={"response_length": len(response)})
        return response, trace_id
    except Exception as e:
        logger.error("RAG query failed for RFX %s: %s", rfx_id, e)
        return f"[ERROR: RAG system failed - {e}]", None


# ============================================================
# Full Evaluation Pipeline
# ============================================================
def evaluate_rfx(ground_truth_file: str) -> dict:
    """Evaluates all Q&A pairs for a single RFX."""
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    rfx_id = gt_data["rfx_id"]
    rfx_name = gt_data["rfx_name"]
    questions = gt_data.get("questions", [])

    if not questions:
        logger.warning("No questions found in %s, skipping.", ground_truth_file)
        return {"rfx_id": rfx_id, "rfx_name": rfx_name, "evaluations": [], "skipped": True}

    logger.info("Evaluating RFX: %s (%s) - %d questions", rfx_name, rfx_id, len(questions))

    evaluations = []
    for i, qa in enumerate(questions, 1):
        question = qa["question"]
        ground_truth = qa["answer"]
        evidence = qa.get("evidence", "")
        category = qa.get("category", "unknown")

        logger.info("  [%d/%d] Category: %s - %s", i, len(questions), category, question[:60])

        # Get RAG response
        rag_response, trace_id = get_rag_response(question, rfx_id)
        time.sleep(1)  # Rate limiting for RAG queries

        # Judge the response
        judgment = evaluate_single_response(question, ground_truth, rag_response, evidence)
        time.sleep(1)  # Rate limiting for judge

        # Push scores to Langfuse trace
        if trace_id and judgment.get("overall_score", 0) > 0:
            for criterion, criterion_score in judgment.get("scores", {}).items():
                langfuse.create_score(
                    trace_id=trace_id,
                    name=criterion,
                    value=float(criterion_score),
                    data_type="NUMERIC",
                    comment=judgment.get("reasoning", "")[:200],
                    metadata={"rfx_id": rfx_id, "category": category},
                )
            langfuse.create_score(
                trace_id=trace_id,
                name="overall_score",
                value=float(judgment["overall_score"]),
                data_type="NUMERIC",
                comment=judgment.get("reasoning", "")[:200],
                metadata={"rfx_id": rfx_id, "category": category},
            )

        evaluation = {
            "question_id": qa.get("id", i),
            "category": category,
            "question": question,
            "ground_truth_answer": ground_truth,
            "rag_response": rag_response,
            "judgment": judgment,
        }
        evaluations.append(evaluation)

    # Calculate aggregate scores
    valid_evals = [e for e in evaluations if e["judgment"]["overall_score"] > 0]
    if valid_evals:
        avg_scores = {}
        for criterion in EVALUATION_CRITERIA:
            scores = [e["judgment"]["scores"].get(criterion, 0) for e in valid_evals]
            avg_scores[criterion] = round(sum(scores) / len(scores), 2)

        overall_scores = [e["judgment"]["overall_score"] for e in valid_evals]
        avg_overall = round(sum(overall_scores) / len(overall_scores), 2)
    else:
        avg_scores = {k: 0 for k in EVALUATION_CRITERIA}
        avg_overall = 0

    # Category breakdown
    category_scores = {}
    for e in valid_evals:
        cat = e["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(e["judgment"]["overall_score"])

    category_averages = {
        cat: round(sum(scores) / len(scores), 2)
        for cat, scores in category_scores.items()
    }

    result = {
        "rfx_id": rfx_id,
        "rfx_name": rfx_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "evaluated": len(valid_evals),
        "errors": len(evaluations) - len(valid_evals),
        "average_scores": avg_scores,
        "average_overall_score": avg_overall,
        "category_averages": category_averages,
        "evaluations": evaluations,
    }

    return result


def generate_report(results: list[dict]) -> dict:
    """Generates an aggregate evaluation report across all RFXs."""
    all_scores = {k: [] for k in EVALUATION_CRITERIA}
    all_overall = []
    all_category_scores = {}
    total_questions = 0
    total_evaluated = 0
    total_errors = 0

    for result in results:
        if result.get("skipped"):
            continue
        total_questions += result.get("total_questions", 0)
        total_evaluated += result.get("evaluated", 0)
        total_errors += result.get("errors", 0)

        for criterion, score in result.get("average_scores", {}).items():
            if score > 0:
                all_scores[criterion].append(score)

        if result.get("average_overall_score", 0) > 0:
            all_overall.append(result["average_overall_score"])

        for cat, avg in result.get("category_averages", {}).items():
            if cat not in all_category_scores:
                all_category_scores[cat] = []
            all_category_scores[cat].append(avg)

    report = {
        "report_timestamp": datetime.now().isoformat(),
        "total_rfx_evaluated": len([r for r in results if not r.get("skipped")]),
        "total_questions": total_questions,
        "total_evaluated": total_evaluated,
        "total_errors": total_errors,
        "global_average_scores": {
            k: round(sum(v) / len(v), 2) if v else 0
            for k, v in all_scores.items()
        },
        "global_overall_score": round(sum(all_overall) / len(all_overall), 2) if all_overall else 0,
        "category_performance": {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in all_category_scores.items()
            if scores
        },
        "per_rfx_summary": [
            {
                "rfx_id": r["rfx_id"],
                "rfx_name": r["rfx_name"],
                "overall_score": r.get("average_overall_score", 0),
                "questions_evaluated": r.get("evaluated", 0),
            }
            for r in results
            if not r.get("skipped")
        ],
    }

    return report


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation Framework")
    parser.add_argument("--rfx-id", type=str, help="Evaluate a single RFX by ID")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated without running")
    args = parser.parse_args()

    # Find ground truth files
    gt_files = [
        os.path.join(GROUND_TRUTH_DIR, f)
        for f in os.listdir(GROUND_TRUTH_DIR)
        if f.endswith(".json") and not f.startswith("_")
    ]

    if not gt_files:
        logger.error("No ground truth files found in %s. Run generate_ground_truth.py first.", GROUND_TRUTH_DIR)
        return

    # Filter by RFX ID if specified
    if args.rfx_id:
        gt_files = [f for f in gt_files if args.rfx_id in os.path.basename(f)]
        if not gt_files:
            logger.error("No ground truth file found for RFX ID: %s", args.rfx_id)
            return

    if args.dry_run:
        logger.info("DRY RUN - Would evaluate %d ground truth files:", len(gt_files))
        for f in gt_files:
            logger.info("  - %s", os.path.basename(f))
        return

    if args.report:
        # Load existing results
        result_files = [
            os.path.join(RESULTS_DIR, f)
            for f in os.listdir(RESULTS_DIR)
            if f.startswith("eval_rfx_") and f.endswith(".json")
        ]
        results = []
        for rf in result_files:
            with open(rf, "r", encoding="utf-8") as f:
                results.append(json.load(f))

        report = generate_report(results)
        report_path = os.path.join(RESULTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to: %s", report_path)
        print(json.dumps(report, indent=2))
        return

    # Run evaluation
    logger.info("=" * 60)
    logger.info("LLM-AS-A-JUDGE EVALUATION")
    logger.info("Ground truth files: %d", len(gt_files))
    logger.info("=" * 60)

    results = []
    for i, gt_file in enumerate(gt_files, 1):
        logger.info("[%d/%d] Evaluating: %s", i, len(gt_files), os.path.basename(gt_file))
        try:
            result = evaluate_rfx(gt_file)
            results.append(result)

            # Save individual result
            result_filename = f"eval_rfx_{result['rfx_id']}.json"
            result_path = os.path.join(RESULTS_DIR, result_filename)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info("  Score: %.1f/10 | Saved: %s", result.get("average_overall_score", 0), result_filename)

        except Exception as e:
            logger.error("  FAILED: %s", e)

        if i < len(gt_files):
            time.sleep(2)

    # Generate aggregate report
    report = generate_report(results)
    report_path = os.path.join(RESULTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("  Global Score: %.1f/10", report["global_overall_score"])
    logger.info("  Criterion Scores:")
    for criterion, score in report["global_average_scores"].items():
        logger.info("    %s: %.1f/10", criterion, score)
    logger.info("  Report: %s", report_path)
    logger.info("=" * 60)

    # Flush all Langfuse traces and scores
    langfuse.flush()
    logger.info("Langfuse traces and scores flushed.")


if __name__ == "__main__":
    main()
