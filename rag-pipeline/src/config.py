import json
import os

# Load local.settings.json for local development (no-op in Azure where env vars are already set)
_settings_path = os.path.join(os.path.dirname(__file__), "..", "local.settings.json")
if os.path.exists(_settings_path):
    with open(_settings_path) as _f:
        for _k, _v in json.load(_f).get("Values", {}).items():
            os.environ.setdefault(_k, _v)

# ============================================================
# Azure AI Foundry — Chat Model (GPT-4o-mini)
# ============================================================
# Azure OpenAI deployment endpoint
# Format: https://<resource>.cognitiveservices.azure.com/openai/deployments/<model>
# Swap endpoint URL and key to use a different model — no code changes needed
AI_FOUNDRY_CHAT_ENDPOINT = os.environ["AI_FOUNDRY_CHAT_ENDPOINT"]
AI_FOUNDRY_CHAT_KEY = os.environ["AI_FOUNDRY_CHAT_KEY"]
AI_FOUNDRY_CHAT_MODEL = os.environ.get("AI_FOUNDRY_CHAT_MODEL", "gpt-4o-mini")

# ============================================================
# Azure AI Foundry — Embedding Model
# ============================================================
# Deploy an embedding model from the AI Foundry model catalog
# as a separate serverless endpoint (check North Europe availability)
AI_FOUNDRY_EMBED_ENDPOINT = os.environ["AI_FOUNDRY_EMBED_ENDPOINT"]
AI_FOUNDRY_EMBED_KEY = os.environ["AI_FOUNDRY_EMBED_KEY"]
AI_FOUNDRY_EMBED_MODEL = os.environ.get("AI_FOUNDRY_EMBED_MODEL", "Cohere-embed-v3-multilingual")

# ============================================================
# Azure Cosmos DB (NoSQL with vector search)
# ============================================================
COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
COSMOS_KEY = os.environ["COSMOS_KEY"]
COSMOS_DATABASE = os.environ.get("COSMOS_DATABASE", "rfx-rag")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "rfx-vectors")
COSMOS_CHANGELOG_CONTAINER = os.environ.get("COSMOS_CHANGELOG_CONTAINER", "rfx-changelog")

# Dimensions must match your deployed embedding model.
# Update this value after deploying your embedding endpoint.
VECTOR_DIMENSIONS = int(os.environ.get("VECTOR_DIMENSIONS", "1024"))
TOP_K = int(os.environ.get("TOP_K", "20"))

# ============================================================
# RFX API
# ============================================================
RFX_API_SUBSCRIPTION_KEY = os.environ.get("RFX_API_SUBSCRIPTION_KEY", "")
RFX_API_BASE_URL = os.environ.get("RFX_API_BASE_URL", "")

# ============================================================
# Langfuse — Observability & Tracing
# ============================================================
# Get keys from Langfuse project settings: https://cloud.langfuse.com
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# Set Langfuse env vars so the SDK auto-configures
if LANGFUSE_SECRET_KEY:
    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_BASE_URL", LANGFUSE_BASE_URL)
