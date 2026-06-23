import os
import json
import time
import base64
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from azure.storage.queue import QueueClient, BinaryBase64DecodePolicy, BinaryBase64EncodePolicy

# --- Configuration ---
load_dotenv()
DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
INBOUND_QUEUE = os.getenv("AI_ENGINE_QUESTION_QUEUE", "ai-engine-question-queue")
OUTBOUND_QUEUE = os.getenv("AI_ENGINE_ANSWER_QUEUE", "ai-engine-answer-queue")
QUEUE_POLL_INTERVAL = int(os.getenv("QUEUE_POLL_INTERVAL", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Shared resources (initialized once) ---
llm = ChatOllama(model=MODEL_NAME, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- Auditor prompt (RFX-aware) ---
TEMPLATE = """
You are a professional Procurement AI Auditor. Your job is to extract facts from RFX data with 100% accuracy.

RULES:
1. Use ONLY the provided context. Do not use outside knowledge.
2. Every fact must be linked to the "Supplier" AND the "RFX" mentioned in that specific context chunk.
3. If the context does not contain the answer, state "I cannot find this information in the current data."
4. If you see a conflict between two suppliers, list them separately. Do not merge their capabilities.
5. Never mix facts across different RFXs.

CONTEXT:
{context}

QUESTION: {question}

AUDITOR RESPONSE:"""

PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])


def query_rfx(question: str, rfx_id: str) -> str:
    """
    Core RAG function: queries the vector DB filtered to a single RFX.
    ChromaDB metadata filter ensures only chunks from the opened RFX are retrieved.
    """
    filtered_retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": {"rfx_id": str(rfx_id)}
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=filtered_retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = chain.invoke({"query": question})
    return result["result"]


def decode_queue_message(msg_content: str) -> dict:
    """
    Decodes a queue message. The C# QueueHelper uses Base64 encoding,
    so we need to decode before parsing JSON.
    """
    try:
        decoded = base64.b64decode(msg_content).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        # Fallback: message might already be plain JSON
        return json.loads(msg_content)


def encode_queue_message(payload: dict) -> str:
    """
    Encodes a response message as Base64 JSON to match C# QueueHelper's
    BinaryBase64DecodePolicy expectation.
    """
    json_str = json.dumps(payload)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


# ==========================================================
# MODE 1: Azure Storage Queue Listener (Production)
# ==========================================================
def start_queue_listener():
    """
    Polls the ai-engine-question-queue for messages from the C# Function,
    runs RAG filtered by QuestionnaireId (= RFX ID),
    and sends the answer to ai-engine-answer-queue.
    """
    logger.info(f"Connecting to queues: IN={INBOUND_QUEUE}, OUT={OUTBOUND_QUEUE}")

    inbound = QueueClient.from_connection_string(AZURE_CONN_STR, INBOUND_QUEUE)
    outbound = QueueClient.from_connection_string(AZURE_CONN_STR, OUTBOUND_QUEUE)

    # Ensure queues exist
    try:
        inbound.create_queue()
    except Exception:
        pass
    try:
        outbound.create_queue()
    except Exception:
        pass

    logger.info(f"AI Engine listening (model: {MODEL_NAME}, polling: {QUEUE_POLL_INTERVAL}s)...")

    while True:
        messages = inbound.receive_messages(max_messages=1, visibility_timeout=120)

        for msg in messages:
            try:
                payload = decode_queue_message(msg.content)
                question = payload.get("Question", "")
                rfx_id = str(payload.get("QuestionnaireId", ""))
                company_id = payload.get("CompanyId", 0)
                conversation_id = payload.get("ConversationId", 0)
                question_id = payload.get("QuestionId", 0)

                if not question or not rfx_id:
                    logger.warning(f"Invalid message (missing Question or QuestionnaireId): {payload}")
                    inbound.delete_message(msg)
                    continue

                logger.info(f"[RFX {rfx_id}] Q#{question_id}: {question}")

                # Run RAG filtered to this RFX only
                answer = query_rfx(question, rfx_id)
                logger.info(f"[RFX {rfx_id}] Answer generated ({len(answer)} chars)")

                # Send answer back — matching AiEngineAnswerResponse C# model
                response_payload = encode_queue_message({
                    "CompanyId": company_id,
                    "QuestionnaireId": int(rfx_id),
                    "ConversationId": conversation_id,
                    "QuestionId": question_id,
                    "Answer": answer
                })
                outbound.send_message(response_payload)
                logger.info(f"[RFX {rfx_id}] Answer sent to {OUTBOUND_QUEUE}")

                # Remove processed message
                inbound.delete_message(msg)

            except Exception as e:
                logger.error(f"Failed to process message: {e}", exc_info=True)
                # Message becomes visible again after visibility_timeout

        time.sleep(QUEUE_POLL_INTERVAL)


# ==========================================================
# MODE 2: Interactive CLI (Local Testing)
# ==========================================================
def start_cli():
    """Interactive CLI for testing the retriever locally with RFX filtering."""
    print(f"--- Procurement RAG Brain Active (Model: {MODEL_NAME}) ---")
    print("Commands: 'switch <rfx_id>' to change RFX, 'exit' to quit.\n")

    rfx_id = input("Enter RFX ID (QuestionnaireId) to scope queries: ").strip()

    while True:
        user_query = input("\nQuery: ").strip()

        if user_query.lower() in ['exit', 'quit']:
            break
        if user_query.lower().startswith("switch "):
            rfx_id = user_query.split(" ", 1)[1].strip()
            print(f"Switched to RFX: {rfx_id}")
            continue

        try:
            answer = query_rfx(user_query, rfx_id)
            print(f"\nResponse: {answer}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procurement RAG Retriever")
    parser.add_argument(
        "--mode",
        choices=["queue", "cli"],
        default="cli",
        help="'queue' = Azure Queue listener (production), 'cli' = local testing (default)"
    )
    args = parser.parse_args()

    if args.mode == "queue":
        start_queue_listener()
    else:
        start_cli()