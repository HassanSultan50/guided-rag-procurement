"""
Ground Truth Dataset Generator for LLM-as-a-Judge Evaluation Framework.

This script:
1. Fetches all RFX IDs from the RFX API
2. For each RFX, fetches detailed answering data
3. Uses GPT-4o-mini to generate diverse Q&A pairs grounded in the RFX data
4. Saves one JSON file per RFX in the ground_truth_data/ folder
"""

import json
import os
import sys
import time
import logging

# Setup path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    AI_FOUNDRY_CHAT_ENDPOINT,
    AI_FOUNDRY_CHAT_KEY,
    AI_FOUNDRY_CHAT_MODEL,
    RFX_API_BASE_URL,
    RFX_API_SUBSCRIPTION_KEY,
)
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize chat client
chat_client = ChatCompletionsClient(
    endpoint=AI_FOUNDRY_CHAT_ENDPOINT,
    credential=AzureKeyCredential(AI_FOUNDRY_CHAT_KEY),
    api_version="2025-01-01-preview",
)

# ============================================================
# RFX Data Fetching
# ============================================================
import requests


def fetch_all_rfx_ids():
    """Fetches all RFX IDs from the RFX API."""
    headers = {
        "Ocp-Apim-Subscription-Key": RFX_API_SUBSCRIPTION_KEY,
        "Cache-Control": "no-cache",
    }
    response = requests.get(RFX_API_BASE_URL, headers=headers)
    response.raise_for_status()

    rfx_list = response.json()
    if isinstance(rfx_list, list):
        ids = [str(rfx.get("id") or rfx.get("Id") or rfx.get("rfxId")) for rfx in rfx_list]
    elif isinstance(rfx_list, dict):
        items = rfx_list.get("items") or rfx_list.get("data") or rfx_list.get("rfxs") or []
        ids = [str(item.get("id") or item.get("Id") or item.get("rfxId")) for item in items]
    else:
        ids = []

    return [rid for rid in ids if rid and rid != "None"]


def fetch_rfx_data(rfx_id: str) -> dict:
    """Fetches detailed RFX answering data."""
    url = f"{RFX_API_BASE_URL}/{rfx_id}/answering-data"
    headers = {
        "Ocp-Apim-Subscription-Key": RFX_API_SUBSCRIPTION_KEY,
        "Cache-Control": "no-cache",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


# ============================================================
# RFX Data Summarization (for LLM context)
# ============================================================
def summarize_rfx_for_context(data: dict) -> str:
    """
    Converts raw RFX JSON into a structured text summary suitable for
    passing to the LLM as context for generating Q&A pairs.
    """
    lines = []
    rfx_name = data.get("name", "Unknown RFX")
    rfx_id = str(data.get("id", "N/A"))
    code = data.get("code") or "N/A"
    workspace = data.get("workspaceName") or "N/A"
    starting = (data.get("starting") or "")[:10]
    ending = (data.get("ending") or "")[:10]
    status = data.get("statusExplanation") or "Unknown"

    lines.append(f"RFX: {rfx_name}")
    lines.append(f"ID: {rfx_id} | Code: {code}")
    lines.append(f"Workspace: {workspace}")
    lines.append(f"Period: {starting} to {ending}")
    lines.append(f"Status: {status}")
    lines.append("")

    all_suppliers = data.get("SupplierAnswers", [])
    supplier_names = [s.get("SupplierName", "Unknown") for s in all_suppliers]
    lines.append(f"Suppliers invited ({len(supplier_names)}): {', '.join(supplier_names)}")
    lines.append("")

    # Process each supplier's data
    for supplier in all_suppliers:
        s_name = supplier.get("SupplierName", "Unknown")
        lines.append(f"--- SUPPLIER: {s_name} ---")

        for page in supplier.get("Pages", []):
            page_name = page.get("Name", "")
            lines.append(f"  Page: {page_name}")

            for question in page.get("Questions", []):
                q_name = question.get("Name", "")
                q_type = question.get("Type", 0)
                desc = question.get("Description", "")

                # Read-only buyer info
                if q_type == 10 and desc:
                    lines.append(f"    [Buyer Info] {q_name}: {desc}")
                    continue

                # Section headers
                if q_type == 30:
                    lines.append(f"    [Section] {q_name}")
                    continue

                # Standard answers
                for answer in question.get("Answers", []):
                    text_val = answer.get("TextValue")
                    double_val = answer.get("DoubleValue")
                    if text_val:
                        lines.append(f"    Q: {q_name} -> A: {text_val}")
                    elif double_val is not None:
                        lines.append(f"    Q: {q_name} -> A: {double_val}")

                # Choices
                choices = question.get("Choices", [])
                selected = [c.get("ChoiceName", "") for c in choices if c.get("Selected")]
                if selected:
                    lines.append(f"    Q: {q_name} -> Selected: {', '.join(selected)}")

                # Product table data
                products = {p["RfxProductId"]: p for p in question.get("TableProducts", [])}
                for tpq in question.get("TableProductQuestions", []):
                    tpq_name = tpq.get("Name", "")
                    for ta in tpq.get("Answers", []):
                        prod_id = ta.get("RfxProductId")
                        product = products.get(prod_id, {})
                        p_name = product.get("Name", "Product")
                        batch_size = product.get("BatchSize", "")
                        batch_unit = product.get("BatchUnit", "")

                        text_val = ta.get("TextValue")
                        double_val = ta.get("DoubleValue")
                        val = text_val if text_val else (str(double_val) if double_val is not None else None)

                        if val:
                            batch_info = f" (batch: {batch_size} {batch_unit})" if batch_size else ""
                            lines.append(f"    Product: {p_name}{batch_info} | {tpq_name}: {val}")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# Q&A Generation via LLM
# ============================================================
GENERATION_SYSTEM_PROMPT = """You are an expert evaluation dataset generator for procurement RFX (Request for X) systems.

Your task is to generate high-quality question-answer pairs from the provided RFX data. These Q&A pairs will be used as ground truth to evaluate a RAG (Retrieval Augmented Generation) system.

RULES:
1. Every question MUST be answerable strictly from the provided RFX data
2. Every answer MUST be factually correct and grounded in the data — NO hallucination
3. Answers should be detailed and include specific data points (names, numbers, dates)
4. Generate diverse questions across ALL categories listed below
5. Questions should be realistic — things a procurement professional would actually ask
6. If the RFX has limited data, generate fewer but higher quality questions

QUESTION CATEGORIES (generate at least 2-3 per category where data permits):

1. FACTUAL: Direct fact retrieval (supplier names, dates, status, codes)
   Example: "Which suppliers were invited to this RFX?"

2. REASONING: Require inference or logical deduction from the data
   Example: "Based on the responses, which supplier appears most qualified?"

3. COMPARISON: Compare suppliers, prices, or offerings
   Example: "How do the prices from Supplier A compare to Supplier B for Product X?"

4. ANALYTICAL/CALCULATION: Require computation or data analysis
   Example: "What is the total cost for Product X from Supplier A given the batch size?"

5. COMPLETENESS: About data gaps, missing responses, or participation
   Example: "Which suppliers did not respond to this RFX?"

6. COMPLIANCE: About certifications, requirements, or standards adherence
   Example: "Which suppliers have ISO 9001 certification?"

7. CONTEXTUAL: About the RFX structure, timeline, or scope
   Example: "What is the time period covered by this RFX?"

8. SYNTHESIS: Require combining information from multiple parts of the data
   Example: "Summarize the key differences between the top two suppliers."

OUTPUT FORMAT (strict JSON):
{
  "questions": [
    {
      "id": 1,
      "category": "factual|reasoning|comparison|analytical|completeness|compliance|contextual|synthesis",
      "question": "The question text",
      "answer": "The detailed answer grounded in the data",
      "evidence": "The specific data points from the RFX that support this answer"
    }
  ]
}

Generate between 10-20 questions per RFX depending on data richness. Ensure good coverage across categories."""


def generate_qa_pairs(rfx_summary: str, rfx_name: str, rfx_id: str) -> dict:
    """Uses GPT-4o-mini to generate Q&A pairs from RFX data."""
    # Truncate context if too large (128K token limit ~ 300K chars conservatively)
    max_chars = 300000
    if len(rfx_summary) > max_chars:
        rfx_summary = rfx_summary[:max_chars] + "\n\n[... DATA TRUNCATED DUE TO SIZE ...]"

    user_prompt = f"""Generate ground truth question-answer pairs from this RFX data:

RFX Name: {rfx_name}
RFX ID: {rfx_id}

=== RFX DATA START ===
{rfx_summary}
=== RFX DATA END ===

Generate diverse, high-quality Q&A pairs covering all applicable categories. 
Return ONLY valid JSON matching the specified format."""

    try:
        response = chat_client.complete(
            messages=[
                SystemMessage(content=GENERATION_SYSTEM_PROMPT),
                UserMessage(content=user_prompt),
            ],
            model=AI_FOUNDRY_CHAT_MODEL,
            temperature=0.3,
            max_tokens=4000,
        )

        content = response.choices[0].message.content.strip()

        # Handle markdown code blocks in response
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON for RFX %s: %s", rfx_id, e)
        logger.debug("Raw response: %s", content[:500] if 'content' in dir() else "N/A")
        return {"questions": [], "error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        logger.error("LLM generation failed for RFX %s: %s", rfx_id, e)
        return {"questions": [], "error": str(e)}


# ============================================================
# Main Pipeline
# ============================================================
def process_single_rfx(rfx_id: str) -> dict:
    """Fetches RFX data, generates Q&A pairs, and saves to file."""
    logger.info("Fetching data for RFX %s...", rfx_id)
    data = fetch_rfx_data(rfx_id)

    rfx_name = data.get("name", f"RFX_{rfx_id}")
    logger.info("Processing: %s (ID: %s)", rfx_name, rfx_id)

    # Build text summary for LLM context
    rfx_summary = summarize_rfx_for_context(data)

    # Generate Q&A pairs
    logger.info("Generating Q&A pairs for RFX %s...", rfx_id)
    qa_data = generate_qa_pairs(rfx_summary, rfx_name, rfx_id)

    # Build output document
    output = {
        "rfx_id": rfx_id,
        "rfx_name": rfx_name,
        "rfx_code": data.get("code", "N/A"),
        "rfx_period": {
            "start": (data.get("starting") or "")[:10],
            "end": (data.get("ending") or "")[:10],
        },
        "suppliers_invited": [s.get("SupplierName", "Unknown") for s in data.get("SupplierAnswers", [])],
        "total_questions_generated": len(qa_data.get("questions", [])),
        "questions": qa_data.get("questions", []),
    }

    if "error" in qa_data:
        output["generation_error"] = qa_data["error"]

    # Save to file
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in rfx_name).strip()
    filename = f"rfx_{rfx_id}_{safe_name[:50]}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d Q&A pairs to: %s", len(output["questions"]), filename)
    return output


def main():
    """Main entry point — processes all RFXs."""
    logger.info("=" * 60)
    logger.info("GROUND TRUTH DATASET GENERATION")
    logger.info("=" * 60)

    # Fetch all RFX IDs
    rfx_ids = fetch_all_rfx_ids()
    logger.info("Found %d RFXs to process", len(rfx_ids))

    results = {"total_rfx": len(rfx_ids), "processed": 0, "failed": 0, "total_questions": 0}
    failed_ids = []

    for i, rfx_id in enumerate(rfx_ids, 1):
        logger.info("[%d/%d] Processing RFX: %s", i, len(rfx_ids), rfx_id)
        try:
            output = process_single_rfx(rfx_id)
            results["processed"] += 1
            results["total_questions"] += output["total_questions_generated"]
        except Exception as e:
            logger.error("[%d/%d] FAILED for RFX %s: %s", i, len(rfx_ids), rfx_id, e)
            results["failed"] += 1
            failed_ids.append(rfx_id)

        # Rate limiting
        if i < len(rfx_ids):
            time.sleep(2)

    # Save summary
    results["failed_ids"] = failed_ids
    summary_path = os.path.join(OUTPUT_DIR, "_generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("  Processed: %d/%d", results["processed"], results["total_rfx"])
    logger.info("  Failed: %d", results["failed"])
    logger.info("  Total Q&A pairs: %d", results["total_questions"])
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
