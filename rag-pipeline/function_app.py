import azure.functions as func
import json
import base64
import logging
from langfuse import get_client
from src.retriever import query_rfx
from src.reingest import reingest_rfx

app = func.FunctionApp()
langfuse = get_client()


@app.queue_trigger(
    arg_name="azqueue",
    queue_name="ai-questionnaire-conversation-question-create-queue",
    connection="AiCommunicationStorage",
)
@app.queue_output(
    arg_name="msg_out",
    queue_name="ai-questionnaire-conversation-question-answer-queue",
    connection="AiCommunicationStorage",
)
def ai_engine_process_question(azqueue: func.QueueMessage, msg_out: func.Out[str]):
    body = azqueue.get_body().decode("utf-8")

    # Decode — C# QueueHelper uses Base64 encoding
    try:
        payload = json.loads(base64.b64decode(body).decode("utf-8"))
    except Exception:
        payload = json.loads(body)

    question = payload.get("Question", "")
    rfx_id = str(payload.get("QuestionnaireId", ""))
    role = payload.get("UserRole", "default")
    logging.info("[RFX %s] Processing question: %s...", rfx_id, question[:80])

    if not question or not rfx_id:
        logging.warning("Invalid message — missing Question or QuestionnaireId: %s", payload)
        return

    answer = query_rfx(question, rfx_id, role=role)
    logging.info("[RFX %s] Answer generated (%d chars)", rfx_id, len(answer))

    # Flush Langfuse traces before the function exits
    langfuse.flush()

    # Build response matching AiEngineAnswerResponse C# model
    response = json.dumps({
        "CompanyId": payload.get("CompanyId", 0),
        "QuestionnaireId": int(rfx_id),
        "ConversationId": payload.get("ConversationId", 0),
        "QuestionId": payload.get("QuestionId", 0),
        "Answer": answer,
    })

    msg_out.set(response)
    logging.info("[RFX %s] Response enqueued to answer queue.", rfx_id)


@app.queue_trigger(
    arg_name="azqueue",
    queue_name="ai-questionnaire-ingestion-queue",
    connection="AiCommunicationStorage",
)
@app.queue_output(
    arg_name="msg_out",
    queue_name="ai-questionnaire-ingestion-response-queue",
    connection="AiCommunicationStorage",
)
def ai_engine_reingest_rfx(azqueue: func.QueueMessage, msg_out: func.Out[str]):
    """
    Triggered when a supplier reopens and updates an RFX.
    Deletes stale embeddings, fetches latest data, re-ingests,
    and logs changes to the changelog container.
    """
    body = azqueue.get_body().decode("utf-8")

    # Decode — C# QueueHelper uses Base64 encoding
    try:
        payload = json.loads(base64.b64decode(body).decode("utf-8"))
    except Exception:
        payload = json.loads(body)

    rfx_id = str(payload.get("QuestionnaireId", ""))
    company_id = payload.get("CompanyId", 0)
    logging.info("[RFX %s] Re-ingestion triggered (CompanyId: %d)", rfx_id, company_id)

    if not rfx_id:
        logging.warning("Invalid message — missing QuestionnaireId: %s", payload)
        return

    try:
        result = reingest_rfx(rfx_id)
        logging.info(
            "[RFX %s] Re-ingestion complete. Old: %d docs, New: %d docs. Changes: +%d -%d ~%d",
            rfx_id,
            result["old_doc_count"],
            result["new_doc_count"],
            result["changes"]["added_count"],
            result["changes"]["removed_count"],
            result["changes"]["modified_count"],
        )
    except Exception as e:
        logging.error("[RFX %s] Re-ingestion failed: %s", rfx_id, e)
        raise
    finally:
        # Flush Langfuse traces before the function exits
        langfuse.flush()

    # Send response to confirm completion
    response = json.dumps({
        "QuestionnaireId": int(rfx_id),
    })
    msg_out.set(response)
    logging.info("[RFX %s] Ingestion response enqueued.", rfx_id)