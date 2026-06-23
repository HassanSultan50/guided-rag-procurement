import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Paths
DATA_FILE = "data/rfx_detailed.json"
DB_DIR = "vector_db"

def parse_rfx_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip the HTTP header lines
        content = f.read()
        json_start = content.find('{')
        data = json.loads(content[json_start:])

    documents = []
    rfx_name = data.get("name", "Unknown RFX")
    
    # Loop through each supplier's answers
    for supplier in data.get("SupplierAnswers", []):
        s_name = supplier.get("SupplierName")
        for page in supplier.get("Pages", []):
            for question in page.get("Questions", []):
                q_name = question.get("Name")
                
                # Handle standard text answers
                for answer in question.get("Answers", []):
                    text_val = answer.get("TextValue")
                    if text_val:
                        content = f"RFX: {rfx_name}\nSupplier: {s_name}\nQuestion: {q_name}\nAnswer: {text_val}"
                        documents.append(Document(page_content=content, metadata={"supplier": s_name}))
                
                # Handle multiple-choice (Certifications)
                selected_choices = [c.get("ChoiceName") for c in question.get("Choices", []) if c.get("Selected")]
                if selected_choices:
                    choices_text = ", ".join(selected_choices)
                    content = f"RFX: {rfx_name}\nSupplier: {s_name}\nRequirement: {q_name}\nSelected Options: {choices_text}"
                    documents.append(Document(page_content=content, metadata={"supplier": s_name}))
    return documents

def run_ingestion():
    print("--- Starting Local Ingestion ---")
    docs = parse_rfx_json(DATA_FILE)
    
    print(f"Loaded {len(docs)} context-aware chunks.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"Ingestion complete! Database saved to {DB_DIR}")

if __name__ == "__main__":
    run_ingestion()