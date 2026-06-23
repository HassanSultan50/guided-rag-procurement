import os
import requests
import json
import logging
from src import Document
from src.config import RFX_API_BASE_URL, RFX_API_SUBSCRIPTION_KEY
from src.ingestion import ingest_documents

logger = logging.getLogger(__name__)

# Phrases that indicate a supplier has declined rather than simply not responding
_DECLINE_PHRASES = [
    "does not provide", "do not provide", "cannot provide",
    "not able to", "we decline", "not our area", "not in our scope",
    "we don't offer", "we do not offer",
]


def fetch_live_rfx_data(rfx_id):
    """Fetches detailed RFX data from Azure APIM."""
    url = f"{RFX_API_BASE_URL}/{rfx_id}/answering-data"

    headers = {
        'Ocp-Apim-Subscription-Key': RFX_API_SUBSCRIPTION_KEY,
        'Cache-Control': 'no-cache'
    }

    logger.info("Connecting to: %s", url)
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        logger.info("Successfully fetched live data.")
        return response.json()
    else:
        try:
            error_msg = response.json().get('message', response.text)
        except Exception:
            error_msg = response.text

        logger.error("Failed to fetch data. Status: %d, Error: %s", response.status_code, error_msg)
        response.raise_for_status()


# ============================================================
# Supplier data-presence detection
# ============================================================
def _supplier_has_data(supplier: dict) -> bool:
    """Returns True if a supplier has any actual response data."""
    for page in supplier.get("Pages", []):
        for question in page.get("Questions", []):
            for a in question.get("Answers", []):
                if a.get("TextValue") or a.get("DoubleValue") is not None:
                    return True
            for tpq in question.get("TableProductQuestions", []):
                for ta in tpq.get("Answers", []):
                    if ta.get("TextValue") or ta.get("DoubleValue") is not None:
                        return True
            for c in question.get("Choices", []):
                if c.get("Selected"):
                    return True
    return False


def _supplier_declined(supplier: dict) -> bool:
    """Returns True if a supplier's text answers indicate they declined to bid."""
    for page in supplier.get("Pages", []):
        for question in page.get("Questions", []):
            for a in question.get("Answers", []):
                tv = (a.get("TextValue") or "").lower()
                if any(phrase in tv for phrase in _DECLINE_PHRASES):
                    return True
    return False


# ============================================================
# Co-located pricing documents
# ============================================================
def _build_pricing_summary(supplier_name: str, rfx_label: str, base_meta: dict,
                           question: dict) -> list[Document]:
    """
    Builds consolidated pricing documents per supplier, per product.
    Each document co-locates: price + description + batch size + total cost.
    """
    documents = []
    products = {p["RfxProductId"]: p for p in question.get("TableProducts", [])}

    # Collect all answers grouped by product
    product_data: dict[int, dict] = {}
    for table_q in question.get("TableProductQuestions", []):
        t_q_name = table_q.get("Name", "")
        for table_ans in table_q.get("Answers", []):
            prod_id = table_ans.get("RfxProductId")
            if prod_id not in product_data:
                product_data[prod_id] = {}

            val = table_ans.get("TextValue")
            dval = table_ans.get("DoubleValue")
            if val:
                product_data[prod_id][t_q_name] = val
            if dval is not None:
                product_data[prod_id][f"{t_q_name}_numeric"] = dval

    for prod_id, answers in product_data.items():
        product = products.get(prod_id, {})
        p_name = product.get("Name", "General Service")
        batch_size = product.get("BatchSize")
        batch_unit = product.get("BatchUnit", "")

        parts = [f"{rfx_label} | Supplier: {supplier_name} | Service: {p_name}"]

        for key, val in answers.items():
            if key.endswith("_numeric"):
                continue
            parts.append(f"  {key}: {val}")

        if batch_size:
            parts.append(f"  Batch Size: {batch_size} {batch_unit}".strip())
            for key in answers:
                if "price" in key.lower() and f"{key}_numeric" in answers:
                    try:
                        total = float(answers[f"{key}_numeric"]) * float(batch_size)
                        parts.append(f"  Total Cost ({key}): {total:.2f}")
                    except (ValueError, TypeError):
                        pass

        content = "\n".join(parts)
        documents.append(Document(
            page_content=content,
            metadata={**base_meta, "supplier": supplier_name, "type": "product_spec"}
        ))

    return documents


# ============================================================
# RFX Context Document — the pre-built briefing
# ============================================================
def _build_rfx_context_document(data: dict, rfx_id: str, rfx_name: str,
                                base_meta: dict,
                                responded: list[str],
                                declined: list[str],
                                not_responded: list[str]) -> Document:
    """
    Builds a single, comprehensive RFX Context Document that serves as the
    "briefing" for every query. This document is always injected into context
    regardless of query type or vector search results.

    It contains:
    - RFX identification and metadata
    - Questionnaire structure (pages, questions, types)
    - Products/services being quoted (with volumes)
    - Supplier participation summary
    - Data quality observations
    """
    lines = [
        f"RFX CONTEXT DOCUMENT: {rfx_name} (ID: {rfx_id})",
        "",
    ]

    # RFX metadata
    code = data.get("code") or "N/A"
    workspace = data.get("workspaceName") or "N/A"
    starting = (data.get("starting") or "")[:10]
    ending = (data.get("ending") or "")[:10]
    status = data.get("statusExplanation") or "Unknown"

    lines.append(f"Code: {code} | Workspace: {workspace}")
    lines.append(f"Period: {starting} to {ending} | Status: {status}")
    lines.append("")

    # Questionnaire structure
    all_suppliers = data.get("SupplierAnswers", [])
    first = all_suppliers[0] if all_suppliers else {}

    lines.append("QUESTIONNAIRE STRUCTURE:")
    products_info = []
    for page in first.get("Pages", []):
        p_name = page.get("Name", "")
        lines.append(f"  Page: {p_name}")
        for q in page.get("Questions", []):
            q_name = q.get("Name", "")
            q_type = q.get("Type", 0)
            type_label = {
                10: "Read-Only Info", 30: "Section Header", 40: "Product Table",
                100: "Short Text", 110: "Long Text", 120: "Number",
                130: "Date", 140: "File Upload", 150: "Multi-Select", 160: "Single-Select",
            }.get(q_type, f"Type {q_type}")

            lines.append(f"    - {q_name} ({type_label})")

            # Capture product definitions
            for p in q.get("TableProducts", []):
                batch_str = ""
                if p.get("BatchSize"):
                    batch_str = f", Volume: {p['BatchSize']} {p.get('BatchUnit', '')}"
                products_info.append(f"  - {p.get('Name', '?')}{batch_str}")

            # Capture choice options (e.g. certifications)
            choices = q.get("Choices", [])
            if choices:
                choice_names = [c.get("ChoiceName", "") for c in choices]
                lines.append(f"      Options: {', '.join(choice_names)}")

    if products_info:
        lines.append("")
        lines.append("PRODUCTS/SERVICES BEING QUOTED:")
        lines.extend(products_info)

    # Supplier participation summary
    total = len(all_suppliers)
    lines.append("")
    lines.append(f"SUPPLIER PARTICIPATION ({total} invited):")
    if responded:
        lines.append(f"  Responded ({len(responded)}): {', '.join(responded)}")
    if declined:
        lines.append(f"  Declined to bid ({len(declined)}): {', '.join(declined)}")
    if not_responded:
        lines.append(f"  Did NOT respond ({len(not_responded)}): {', '.join(not_responded)}")

    # Data quality observations
    quality_notes = []
    for s in all_suppliers:
        s_name = s.get("SupplierName", "Unknown")
        for page in s.get("Pages", []):
            for q in page.get("Questions", []):
                for tpq in q.get("TableProductQuestions", []):
                    for ta in tpq.get("Answers", []):
                        tv = ta.get("TextValue", "")
                        if tv and len(tv.strip()) <= 1:
                            quality_notes.append(
                                f"  - {s_name}: Single-character answer in product table (possible test/garbage data)")
                            break
                    if quality_notes:
                        break
                # Check for identical descriptions
                descs = []
                for tpq in q.get("TableProductQuestions", []):
                    if "desc" in tpq.get("Name", "").lower():
                        descs = [ta.get("TextValue", "") for ta in tpq.get("Answers", [])]
                        break
                if len(descs) > 2 and len(set(descs)) == 1 and descs[0]:
                    quality_notes.append(
                        f"  - {s_name}: All product descriptions identical ('{descs[0][:50]}') — possible copy-paste")

    if quality_notes:
        lines.append("")
        lines.append("DATA QUALITY OBSERVATIONS:")
        # Deduplicate
        for note in sorted(set(quality_notes)):
            lines.append(note)

    return Document(
        page_content="\n".join(lines),
        metadata={**base_meta, "type": "rfx_context"},
    )


# ============================================================
# Main parser
# ============================================================
def parse_rfx_to_documents(data):
    """
    Converts RFX JSON from the RFX API into indexed Document objects.

    Document types produced:
      1. rfx_context — comprehensive RFX briefing (always injected into LLM context)
      2. participant_list — who responded / declined / didn't respond
      3. requirement — buyer-provided scope of work / requirements
      4. certification — supplier's selected compliance certifications
      5. standard_answer — supplier's text answers to questions
      6. product_spec — co-located price + description + batch size per product per supplier
      7. supplier_summary — one-paragraph summary per supplier
    """
    documents = []
    rfx_name = data.get("name", "Unknown RFX")
    rfx_id = str(data.get("id", "N/A"))

    base_meta = {"rfx_id": rfx_id, "rfx_name": rfx_name}
    rfx_label = f"RFX: {rfx_name} (ID: {rfx_id})"

    all_suppliers = data.get("SupplierAnswers", [])

    # ----------------------------------------------------------------
    # 0. Classify each supplier's participation status
    # ----------------------------------------------------------------
    responded = []
    declined = []
    not_responded = []

    for s in all_suppliers:
        s_name = s.get("SupplierName", "Unknown")
        if not _supplier_has_data(s):
            not_responded.append(s_name)
        elif _supplier_declined(s):
            declined.append(s_name)
        else:
            responded.append(s_name)

    # ----------------------------------------------------------------
    # 1. RFX Context Document (the "briefing")
    # ----------------------------------------------------------------
    rfx_context_doc = _build_rfx_context_document(
        data, rfx_id, rfx_name, base_meta, responded, declined, not_responded
    )
    documents.append(rfx_context_doc)

    # ----------------------------------------------------------------
    # 2. Participant List
    # ----------------------------------------------------------------
    all_names = [s.get("SupplierName", "Unknown") for s in all_suppliers]
    participant_parts = [
        f"{rfx_label} | RFX PARTICIPANT LIST",
        f"Total suppliers invited: {len(all_names)}",
        f"Suppliers who responded: {', '.join(responded) if responded else 'None'}",
        f"Suppliers who declined to bid: {', '.join(declined) if declined else 'None'}",
        f"Suppliers who did NOT respond (no data at all): {', '.join(not_responded) if not_responded else 'None'}",
    ]
    documents.append(Document(
        page_content="\n".join(participant_parts),
        metadata={**base_meta, "type": "participant_list"},
    ))

    # ----------------------------------------------------------------
    # 3. Global Project Requirements (from buyer-provided read-only fields)
    # ----------------------------------------------------------------
    first_supplier = all_suppliers[0] if all_suppliers else {}
    for page in first_supplier.get("Pages", []):
        for question in page.get("Questions", []):
            q_name = question.get("Name", "")
            q_type = question.get("Type", 0)
            desc = question.get("Description")

            # Type 10 = read-only info from the buyer
            if q_type == 10 and desc:
                content = f"{rfx_label} | BUYER INFORMATION | {q_name}: {desc}"
                documents.append(Document(
                    page_content=content,
                    metadata={**base_meta, "type": "requirement"},
                ))

    # ----------------------------------------------------------------
    # 4. Supplier-Specific Data
    # ----------------------------------------------------------------
    for supplier in all_suppliers:
        s_name = supplier.get("SupplierName", "Unknown")
        summary_parts = [f"{rfx_label} | SUPPLIER SUMMARY: {s_name}"]
        has_any = False

        for page in supplier.get("Pages", []):
            for question in page.get("Questions", []):
                q_name = question.get("Name", "")

                # A. Certifications / Multi-select
                selected_choices = [
                    c.get("ChoiceName") for c in question.get("Choices", [])
                    if c.get("Selected")
                ]
                if selected_choices:
                    content = (f"{rfx_label} | Supplier: {s_name} | "
                               f"Verified Certifications: {', '.join(selected_choices)}")
                    documents.append(Document(
                        page_content=content,
                        metadata={**base_meta, "supplier": s_name, "type": "certification"},
                    ))
                    summary_parts.append(f"  Certifications: {', '.join(selected_choices)}")
                    has_any = True

                # B. Standard Text/Number/Date Answers
                for answer in question.get("Answers", []):
                    val = answer.get("TextValue")
                    if val:
                        content = (f"{rfx_label} | Supplier: {s_name} | "
                                   f"Question: {q_name} | Answer: {val}")
                        documents.append(Document(
                            page_content=content,
                            metadata={**base_meta, "supplier": s_name, "type": "standard_answer"},
                        ))
                        summary_parts.append(f"  {q_name}: {val[:100]}")
                        has_any = True

                # C. Product Tables — co-located pricing
                if question.get("TableProductQuestions"):
                    pricing_docs = _build_pricing_summary(
                        s_name, rfx_label, base_meta, question
                    )
                    documents.extend(pricing_docs)
                    if pricing_docs:
                        has_any = True
                        for pd in pricing_docs:
                            first_line = pd.page_content.split("\n")[0]
                            svc = first_line.split("|")[-1].strip()
                            summary_parts.append(f"  {svc}")

        if not has_any:
            summary_parts.append("  STATUS: Did NOT respond to this RFX. No data provided.")

        documents.append(Document(
            page_content="\n".join(summary_parts),
            metadata={**base_meta, "supplier": s_name, "type": "supplier_summary"},
        ))

    return documents


def run_live_ingestion(rfx_id):
    """Fetches an RFX from the API and ingests into Cosmos DB."""
    try:
        raw_json = fetch_live_rfx_data(rfx_id)
        logger.info("Parsing JSON into searchable documents...")
        docs = parse_rfx_to_documents(raw_json)
        logger.info("Created %d context-aware chunks.", len(docs))

        logger.info("Ingesting into Cosmos DB...")
        ingest_documents(docs)
        logger.info("Success! RFX %s is now stored in Cosmos DB.", rfx_id)

    except Exception as e:
        logger.error("Ingestion failed: %s", e)
