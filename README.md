# Guided RAG for Procurement Response Analysis

A Retrieval-Augmented Generation (RAG) pipeline that ingests structured procurement
**RFX** (Request for X — Quote / Proposal / Information) supplier responses, embeds and
indexes them for semantic search, and generates **grounded, hallucination-resistant
comparative summaries** in response to user-guided focus criteria
(e.g. *"compare all suppliers on safety certifications"* or
*"summarise logistics costs across suppliers"*).

> **This repository is my Master's-thesis work, published here as a sanitized,
> code-only version.** The thesis covers guided AI summarisation for the comparative
> analysis of technically complex procurement bids.

> **Author / role:** Muhammad Hassan Sultan — sole developer of the RAG pipeline,
> retrieval logic, prompt design, and evaluation framework.
>
> **Sanitized — scope & data:** The code has been deliberately sanitized for public
> release. It contains **no datasets** of any kind — no source documents, ground-truth,
> embeddings, or experiment outputs — and **no organisation-specific names, endpoints,
> deployment configuration, or credentials**. What remains is generic, reusable RAG
> infrastructure. To run it you supply your own data source and your own service
> credentials via environment variables.

---

## What it does

- **Ingests** structured supplier questionnaire responses into semantically coherent
  document units (ontology-driven decomposition rather than naive fixed-size chunking).
- **Embeds** documents with **Cohere multilingual embeddings** (1024-dim).
- **Stores & indexes** them in **Azure Cosmos DB** using **DiskANN** vector indexing.
- **Retrieves** relevant context via hybrid (vector + structural) search, with a
  question-type classification step.
- **Generates** comparative answers with an LLM (GPT-4o-mini / Phi-4 via Azure AI
  Foundry) using domain-specific, abstention-aware prompts.
- **Evaluates** retrieval quality and answer faithfulness with an
  **LLM-as-a-Judge** framework plus precision@K / MRR.

The pipeline was first prototyped locally (ChromaDB + Ollama / Mistral 7B) and then
migrated to a cloud-native Azure deployment; the thesis analyses that architectural
evolution (latency, cost, scalability, accuracy trade-offs).

## Repository layout

| Path | Contents |
|------|----------|
| `rag-pipeline/src/` | Core pipeline: `config`, `domain` ontology, `ingestion`, `retriever`, `changelog`, re-ingestion |
| `rag-pipeline/*.py` | Experiment & evaluation drivers (`evaluate_rag`, `run_rag_experiment`, `run_judge_evaluation`, `generate_ground_truth`, Langfuse upload) |
| `rag-pipeline/function_app.py` | Azure Functions queue-trigger entry points |
| `ingestion-engine/src/` | Earlier local-prototype ingestion/retrieval (ChromaDB + HuggingFace embeddings) |

## Configuration

All secrets and endpoints are supplied via environment variables — nothing is committed.
Copy the example settings and fill in your own values:

```bash
cp rag-pipeline/example.local.settings.json rag-pipeline/local.settings.json
# edit local.settings.json with your own service endpoints, keys, and data-source URL
```

Key variables: `AI_FOUNDRY_CHAT_ENDPOINT` / `_KEY`, `AI_FOUNDRY_EMBED_ENDPOINT` / `_KEY`,
`COSMOS_ENDPOINT` / `_KEY`, `VECTOR_DIMENSIONS`, and the data-source variables
`RFX_API_BASE_URL` / `RFX_API_SUBSCRIPTION_KEY`. See `rag-pipeline/src/config.py` for
the full list.

```bash
cd rag-pipeline
python -m pip install -r requirements.txt
```

## Note

This is a personal portfolio extract that demonstrates RAG architecture and evaluation
methodology. It is intentionally generic and ships without any data; point it at your
own RFX/document source to use it.
