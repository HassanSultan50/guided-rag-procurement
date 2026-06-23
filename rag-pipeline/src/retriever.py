import logging
import re
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient
from langfuse import get_client, observe, propagate_attributes
from src import Document
from src.config import (
    AI_FOUNDRY_CHAT_ENDPOINT, AI_FOUNDRY_CHAT_KEY, AI_FOUNDRY_CHAT_MODEL,
    AI_FOUNDRY_EMBED_ENDPOINT, AI_FOUNDRY_EMBED_KEY, AI_FOUNDRY_EMBED_MODEL,
    COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER,
    TOP_K,
)
from src.domain import get_domain_primer, get_role_context

logger = logging.getLogger(__name__)
langfuse = get_client()

# ============================================================
# Clients (initialised once at module level)
# ============================================================
chat_client = ChatCompletionsClient(
    endpoint=AI_FOUNDRY_CHAT_ENDPOINT,
    credential=AzureKeyCredential(AI_FOUNDRY_CHAT_KEY),
    api_version="2025-01-01-preview",
)
embed_client = EmbeddingsClient(
    endpoint=AI_FOUNDRY_EMBED_ENDPOINT,
    credential=AzureKeyCredential(AI_FOUNDRY_EMBED_KEY),
)
_cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
_database = _cosmos_client.get_database_client(COSMOS_DATABASE)
_container = _database.get_container_client(COSMOS_CONTAINER)

# ============================================================
# Query-type classification
# ============================================================
_COMPARISON_KEYWORDS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|rank|ranking|cheapest|most expensive|"
    r"lowest|highest|best|worst|difference|differ|benchmark)\b",
    re.IGNORECASE,
)
_IDENTIFICATION_KEYWORDS = re.compile(
    r"\b(who|which company|which supplier|did not|didn.t|not answer|not respond|"
    r"missing|absent|ignored|no response|no answer|non.respond)\b",
    re.IGNORECASE,
)
_PRICE_KEYWORDS = re.compile(
    r"\b(price|pricing|cost|costs|quote|quotation|rate|rates|budget|spend|"
    r"total cost|unit price|expensive|cheap)\b",
    re.IGNORECASE,
)
_LIST_KEYWORDS = re.compile(
    r"\b(list|create a list|nice list|show all|summarize|summary|overview)\b",
    re.IGNORECASE,
)

# ============================================================
# System prompt core — kept concise; domain knowledge is
# injected dynamically so the prompt stays within limits
# ============================================================
_PROMPT_CORE = """You are a professional Procurement AI Auditor for the RFX system.

RULES:
1. Use ONLY the provided context. Never use outside knowledge.
2. Every fact must be linked to its specific Supplier and RFX.
3. If the context lacks the answer: "I cannot find this information in the current data."
4. Never merge facts across different RFXs or suppliers.
5. A supplier who explicitly declines ("we do not provide this") HAS answered — they "Declined to bid."
6. A supplier with ZERO data in the context "Did not respond to the RFX."
7. When listing prices, include EVERY supplier and EVERY product/service. No omissions.
8. Include Total Cost = unit price × batch size where batch size is available.
9. When asked about a specific supplier, include ALL their data: answers, prices, certifications, descriptions.

OUTPUT:
- Always use Markdown: **bold** names, tables for multi-supplier data, bullets for lists.
- Be concise but COMPLETE — never omit a supplier or product.
- When showing prices, always show: unit price, batch size, and total cost.
- End comparisons with a brief Summary section highlighting key findings."""

_PROMPT_COMPARISON_EXTRA = """
COMPARISON-SPECIFIC RULES:
- Include ALL suppliers. Missing data = "N/A", not omission.
- Present prices in a Markdown table: rows = products, columns = suppliers.
- Include unit, batch size, and Total Cost (unit price × batch size) where available.
- Highlight lowest/highest per item. Flag anomalous pricing (10x+ deviation).
- Flag vague/single-char descriptions and business-domain mismatches.
- End with a "Summary" section noting key differences and recommendation."""

_PROMPT_IDENTIFICATION_EXTRA = """
IDENTIFICATION-SPECIFIC RULES:
- THREE statuses: "Responded with a bid" | "Declined to bid" | "Did not respond"
- "Did not answer" = ZERO data. NOT a supplier who explicitly declined.
- Saying "we don't provide this" = Declined, NOT absent.
- Check the RFX PARTICIPANT LIST for suppliers with no data.
- Categorise EVERY supplier with evidence."""

_PROMPT_PRICING_EXTRA = """
PRICING-SPECIFIC RULES:
- Include ALL suppliers and ALL products — no omissions.
- Markdown table: rows = products, columns = suppliers.
- Show unit, batch size, Total Cost. Missing = "N/A". Declined = "Declined".
- After the table: lowest bid per item, anomalous pricing flags."""


def _classify_query(question: str) -> str:
    """Classifies a user question to select the appropriate prompt and retrieval strategy."""
    q = question.lower()
    if _COMPARISON_KEYWORDS.search(q):
        return "comparison"
    if _IDENTIFICATION_KEYWORDS.search(q):
        return "identification"
    if _PRICE_KEYWORDS.search(q) or _LIST_KEYWORDS.search(q):
        return "pricing"
    return "default"


_QUERY_EXTRAS = {
    "comparison": _PROMPT_COMPARISON_EXTRA,
    "identification": _PROMPT_IDENTIFICATION_EXTRA,
    "pricing": _PROMPT_PRICING_EXTRA,
    "default": "",
}


def _build_system_prompt(query_type: str, role: str = "default") -> str:
    """
    Assembles the system prompt from:
      1. Domain primer — structural knowledge about RFX data (from domain.py)
      2. Role context — user persona adjustments
      3. Core rules — always-on auditor rules
      4. Query-type extra — specialised rules for comparison/identification/pricing
    """
    parts = [
        get_domain_primer(),
        "",
        get_role_context(role),
        "",
        _PROMPT_CORE,
        _QUERY_EXTRAS.get(query_type, ""),
    ]
    return "\n".join(parts)


# ============================================================
# Embedding functions (unchanged — shared with ingestion)
# ============================================================
@observe(name="embed_query")
def embed_query(text: str) -> list[float]:
    """Generates an embedding vector for a single query string."""
    kwargs = {"model": AI_FOUNDRY_EMBED_MODEL} if AI_FOUNDRY_EMBED_MODEL else {}
    try:
        response = embed_client.embed(input=[text], **kwargs)
    except ResourceNotFoundError as ex:
        raise RuntimeError(
            "Embedding deployment was not found. Set AI_FOUNDRY_EMBED_MODEL "
            "in local.settings.json to your embedding deployment/model name."
        ) from ex
    return response.data[0].embedding


@observe(name="embed_documents")
def embed_documents(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Generates embedding vectors for a list of texts in batches."""
    all_embeddings = []
    kwargs = {"model": AI_FOUNDRY_EMBED_MODEL} if AI_FOUNDRY_EMBED_MODEL else {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = embed_client.embed(input=batch, **kwargs)
        except ResourceNotFoundError as ex:
            raise RuntimeError(
                "Embedding deployment was not found. Set AI_FOUNDRY_EMBED_MODEL "
                "in local.settings.json to your embedding deployment/model name."
            ) from ex
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


# ============================================================
# Hybrid retrieval: structured + vector
# ============================================================
@observe(name="fetch_structural_context")
def _fetch_structural_context(rfx_id: str) -> list[Document]:
    """
    Fetches the always-needed structural documents for an RFX:
    - rfx_context (the pre-built RFX briefing document)
    - participant_list
    - supplier_summary (one per supplier)

    These are fetched via metadata filter without vector search,
    ensuring they are ALWAYS included regardless of query embedding similarity.
    """
    structural_types = ["rfx_context", "participant_list", "supplier_summary"]
    placeholders = ", ".join(f"'{t}'" for t in structural_types)
    query = f"""
    SELECT c.id, c.page_content, c.metadata
    FROM c
    WHERE c.metadata.rfx_id = @rfx_id
      AND c.metadata.type IN ({placeholders})
    """
    results = _container.query_items(
        query=query,
        parameters=[{"name": "@rfx_id", "value": str(rfx_id)}],
        enable_cross_partition_query=True,
    )
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in results
    ]


@observe(name="fetch_all_rfx_documents")
def _fetch_all_rfx_documents(rfx_id: str) -> list[Document]:
    """Fetches ALL documents for an RFX from Cosmos DB (no vector search, no top-K limit)."""
    query = """
    SELECT c.id, c.page_content, c.metadata
    FROM c
    WHERE c.metadata.rfx_id = @rfx_id
    """
    results = _container.query_items(
        query=query,
        parameters=[{"name": "@rfx_id", "value": str(rfx_id)}],
        enable_cross_partition_query=True,
    )
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in results
    ]


@observe(name="vector_search")
def _vector_search(query_text: str, rfx_id: str, top_k: int = TOP_K) -> list[Document]:
    """Vector similarity search against Cosmos DB, filtered to a specific RFX."""
    query_vector = embed_query(query_text)

    query = """
    SELECT TOP @top_k
        c.id,
        c.page_content,
        c.metadata,
        VectorDistance(c.embedding, @query_vector) AS score
    FROM c
    WHERE c.metadata.rfx_id = @rfx_id
    ORDER BY VectorDistance(c.embedding, @query_vector)
    """

    results = _container.query_items(
        query=query,
        parameters=[
            {"name": "@top_k", "value": top_k},
            {"name": "@query_vector", "value": query_vector},
            {"name": "@rfx_id", "value": str(rfx_id)},
        ],
        enable_cross_partition_query=True,
    )

    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in results
    ]


@observe(name="hybrid_retrieve")
def _hybrid_retrieve(question: str, rfx_id: str, query_type: str) -> list[Document]:
    """
    Hybrid retrieval strategy:

    1. ALWAYS fetch structural context (rfx_context, participant_list, supplier_summaries)
       — these are the "dedicated context documents" that ground every response.

    2. For exhaustive queries (comparison/identification/pricing):
       — Fetch ALL documents. The structural context is already included.

    3. For default/focused queries:
       — Structural context + vector-searched relevant chunks.
       — Deduplicate by document content to avoid redundancy.
    """
    if query_type in ("comparison", "identification", "pricing"):
        docs = _fetch_all_rfx_documents(rfx_id)
        logger.info("[RFX %s] Exhaustive retrieval: %d docs", rfx_id, len(docs))
        return docs

    # Hybrid: structural + vector
    structural = _fetch_structural_context(rfx_id)
    vector_results = _vector_search(question, rfx_id)

    # Deduplicate: structural docs take priority, add vector results that aren't duplicates
    seen_content = {doc.page_content for doc in structural}
    combined = list(structural)
    for doc in vector_results:
        if doc.page_content not in seen_content:
            combined.append(doc)
            seen_content.add(doc.page_content)

    logger.info(
        "[RFX %s] Hybrid retrieval: %d structural + %d vector = %d total",
        rfx_id, len(structural), len(vector_results), len(combined),
    )
    return combined


# ============================================================
# Token budget management
# ============================================================
_MODEL_MAX_TOKENS = 128_000          # GPT-4o-mini context window
_COMPLETION_TOKENS = 4_096          # max_tokens we request for the response
_TOKEN_BUFFER = 300                 # safety margin
_CHARS_PER_TOKEN = 2.5              # conservative for structured RFX data with whitespace


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~2.5 characters per token for structured RFX data."""
    return int(len(text) / _CHARS_PER_TOKEN)


def _truncate_context(context: str, system_prompt: str) -> str:
    """
    Ensures the total message payload fits within the model's context window.
    Truncation strategy: keep the beginning of the context (which contains
    the RFX briefing and metadata — highest priority) and trim supplier
    data from the end when over budget.
    """
    budget = _MODEL_MAX_TOKENS - _COMPLETION_TOKENS - _TOKEN_BUFFER
    system_tokens = _estimate_tokens(system_prompt)
    # Account for the "CONTEXT:\n...\n\nQUESTION: ...\n\nRESPONSE:" wrapper (~50 tokens)
    available = budget - system_tokens - 50

    if available <= 0:
        return context[:500]  # fallback

    context_tokens = _estimate_tokens(context)
    if context_tokens <= available:
        return context  # fits fine — no truncation needed

    # Truncate to fit: keep from the start (rfx_context and metadata come first)
    max_chars = int(available * _CHARS_PER_TOKEN)
    truncated = context[:max_chars]

    # Try to cut at the last complete supplier section boundary
    last_section = truncated.rfind("\n\n=== SUPPLIER:")
    if last_section > len(truncated) // 2:
        truncated = truncated[:last_section]

    # Count how many suppliers survived truncation
    survived = truncated.count("=== SUPPLIER:")
    total = context.count("=== SUPPLIER:")

    logger.warning(
        "Context truncated from ~%d to ~%d tokens (%d/%d suppliers kept).",
        context_tokens, _estimate_tokens(truncated), survived, total,
    )
    return truncated + f"\n\n[... {total - survived} supplier section(s) omitted due to context length limits ...]"


# ============================================================
# Context assembly
# ============================================================
def _build_context(docs: list[Document], query_type: str) -> str:
    """
    Builds a structured context string from retrieved documents.
    Groups by document type for clarity:
      1. RFX Context (briefing) — always first
      2. RFX Metadata (participant list, requirements)
      3. Supplier sections — grouped by supplier name, sorted by data richness

    Smart filtering:
    - Suppliers with no actual pricing/answer data get a compact summary instead of
      dozens of empty product_spec documents.
    - Suppliers with data are prioritised in the context ordering.
    """
    rfx_context_docs: list[str] = []
    meta_docs: list[str] = []
    supplier_docs: dict[str, list[str]] = {}
    supplier_summaries: dict[str, str] = {}
    supplier_has_data: dict[str, bool] = {}

    for doc in docs:
        doc_type = doc.metadata.get("type", "")
        supplier = doc.metadata.get("supplier")

        if doc_type == "rfx_context":
            rfx_context_docs.append(doc.page_content)
        elif supplier:
            if doc_type == "supplier_summary":
                supplier_summaries[supplier] = doc.page_content
                # Check if this supplier has actual data
                has_data = "Did NOT respond" not in doc.page_content
                supplier_has_data[supplier] = has_data
            supplier_docs.setdefault(supplier, []).append(doc.page_content)
        else:
            meta_docs.append(doc.page_content)

    parts = []

    # 1. RFX briefing document (always first — sets the scene)
    if rfx_context_docs:
        parts.append("=== RFX BRIEFING ===\n" + "\n\n".join(rfx_context_docs))

    # 2. RFX-level metadata
    if meta_docs:
        parts.append("=== RFX METADATA ===\n" + "\n".join(meta_docs))

    # 3. Per-supplier data — prioritise suppliers WITH data
    suppliers_with_data = [s for s in sorted(supplier_docs) if supplier_has_data.get(s, True)]
    suppliers_no_data = [s for s in sorted(supplier_docs) if not supplier_has_data.get(s, True)]

    for supplier_name in suppliers_with_data:
        contents = supplier_docs[supplier_name]
        parts.append(f"=== SUPPLIER: {supplier_name} ===\n" + "\n".join(contents))

    # For non-responding suppliers, include only their summary (not all empty product_specs)
    for supplier_name in suppliers_no_data:
        summary = supplier_summaries.get(supplier_name, "")
        if summary:
            parts.append(f"=== SUPPLIER: {supplier_name} ===\n" + summary)
        else:
            parts.append(f"=== SUPPLIER: {supplier_name} ===\nDid NOT respond to this RFX. No data provided.")

    return "\n\n".join(parts)


# ============================================================
# Core RAG entry point
# ============================================================
@observe(name="query_rfx")
def query_rfx(question: str, rfx_id: str, role: str = "default") -> str:
    """
    Hybrid RAG function:
      1. Classify query type
      2. Build system prompt = domain primer + role context + rules + query-specific rules
      3. Retrieve context = structural docs (always) + vector search (for focused queries)
         OR all docs (for exhaustive queries)
      4. Generate response with the chat model

    Args:
        question: The user's natural-language question.
        rfx_id:   The RFX ID (QuestionnaireId) to scope the search.
        role:     User role for response adaptation (default/procurement_manager/
                  category_specialist/executive).
    """
    query_type = _classify_query(question)
    logger.info("[RFX %s] Query type: %s, Role: %s", rfx_id, query_type, role)

    with propagate_attributes(
        session_id=f"rfx-{rfx_id}",
        metadata={"rfx_id": rfx_id, "query_type": query_type, "role": role},
    ):
        # 1. Retrieve
        docs = _hybrid_retrieve(question, rfx_id, query_type)
        if not docs:
            return "I cannot find this information in the current data."

        # 2. Assemble prompts
        system_prompt = _build_system_prompt(query_type, role)
        context = _build_context(docs, query_type)
        context = _truncate_context(context, system_prompt)
        user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nRESPONSE:"

        # 3. Generate (traced as a generation span)
        chat_kwargs = {"model": AI_FOUNDRY_CHAT_MODEL} if AI_FOUNDRY_CHAT_MODEL else {}
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-generation",
            model=AI_FOUNDRY_CHAT_MODEL,
            input={"system_prompt_length": len(system_prompt), "context_length": len(context), "question": question},
            metadata={"rfx_id": rfx_id, "query_type": query_type, "doc_count": len(docs)},
        ) as generation:
            response = chat_client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_content),
                ],
                temperature=0,
                max_tokens=4096,
                **chat_kwargs,
            )

            answer = response.choices[0].message.content

            # Capture token usage if available
            usage_details = {}
            if hasattr(response, "usage") and response.usage:
                usage_details["input_tokens"] = getattr(response.usage, "prompt_tokens", 0)
                usage_details["output_tokens"] = getattr(response.usage, "completion_tokens", 0)
                usage_details["total_tokens"] = getattr(response.usage, "total_tokens", 0)
            if usage_details:
                generation.update(usage_details=usage_details)

            generation.update(output=answer)

        return answer
