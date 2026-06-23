import os
import requests
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from argparse import ArgumentParser

# Load environment variables (Subscription Key and Base URL)
load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("RFX_API_SUBSCRIPTION_KEY")
BASE_URL = os.getenv("RFX_API_BASE_URL")
DB_DIR = "vector_db"

# def fetch_live_rfx_data(rfx_id):
#     """Fetches detailed RFX data from Azure APIM.""" 
#     url = f"{BASE_URL}/Rfx_GetWithData?id={rfx_id}"
#     headers = {
#         'Ocp-Apim-Subscription-Key': API_KEY,
#         'Cache-Control': 'no-cache'
#     }

#     print(f"Connecting to RFX API for RFX: {rfx_id}...")
#     response = requests.get(url, headers=headers)
    
#     if response.status_code == 200:
#         print("Successfully fetched live data.")
#         return response.json()
#     else:
#         print(f"Failed to fetch data. Status Code: {response.status_code}")
#         print(f"Error Message: {response.text}")
#         response.raise_for_status()

def fetch_live_rfx_data(rfx_id):
    # Constructing the URL: base_url + /id/ + answering-data
    # Result: <RFX_API_BASE_URL>/<rfx_id>/answering-data
    base_url = os.getenv("RFX_API_BASE_URL")
    url = f"{base_url}/{rfx_id}/answering-data"
    
    headers = {
        'Ocp-Apim-Subscription-Key': os.getenv("RFX_API_SUBSCRIPTION_KEY"),
        'Cache-Control': 'no-cache'
    }

    print(f"Connecting to: {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("Successfully fetched live data.")
        return response.json()
    else:
        # APIM often returns JSON error messages
        try:
            error_msg = response.json().get('message', response.text)
        except:
            error_msg = response.text
            
        print(f"Failed to fetch data. Status Code: {response.status_code}")
        print(f"Error: {error_msg}")
        response.raise_for_status()


def parse_rfx_to_documents(data):
    """Converts nested RFX JSON into flat LangChain Document objects with metadata."""
    documents = []
    rfx_name = data.get("name", "Cleaning Services RFX")
    rfx_id = str(data.get("id", "N/A"))

    # Base metadata shared by every document from this RFX
    base_meta = {"rfx_id": rfx_id, "rfx_name": rfx_name}
    rfx_label = f"RFX: {rfx_name} (ID: {rfx_id})"

    # 1. Capture GLOBAL PROJECT REQUIREMENTS
    first_supplier = data.get("SupplierAnswers", [{}])[0]
    for page in first_supplier.get("Pages", []):
        for question in page.get("Questions", []):
            q_name = question.get("Name", "")
            if "Scope of Work" in q_name or "Requirements" in q_name:
                desc = question.get("Description")
                if desc:
                    content = f"{rfx_label} | GLOBAL PROJECT REQUIREMENT | {q_name}: {desc}"
                    documents.append(Document(
                        page_content=content, 
                        metadata={**base_meta, "type": "requirement"}
                    ))

    # 2. Capture SUPPLIER-SPECIFIC DATA
    for supplier in data.get("SupplierAnswers", []):
        s_name = supplier.get("SupplierName")
        
        for page in supplier.get("Pages", []):
            for question in page.get("Questions", []):
                q_name = question.get("Name")
                
                # A. Handle Certifications
                selected_choices = [c.get("ChoiceName") for c in question.get("Choices", []) if c.get("Selected")]
                if selected_choices:
                    content = f"{rfx_label} | Supplier: {s_name} | Verified Certifications: {', '.join(selected_choices)}"
                    documents.append(Document(
                        page_content=content, 
                        metadata={**base_meta, "supplier": s_name, "type": "certification"}
                    ))

                # B. Handle Standard Text Answers
                for answer in question.get("Answers", []):
                    val = answer.get("TextValue")
                    if val:
                        content = f"{rfx_label} | Supplier: {s_name} | Question: {q_name} | Answer: {val}"
                        documents.append(Document(
                            page_content=content, 
                            metadata={**base_meta, "supplier": s_name, "type": "standard_answer"}
                        ))

                # C. Handle PRODUCT TABLES
                for table_q in question.get("TableProductQuestions", []):
                    t_q_name = table_q.get("Name")
                    for table_ans in table_q.get("Answers", []):
                        prod_id = table_ans.get("RfxProductId")
                        product = next((p for p in question.get("TableProducts", []) if p["RfxProductId"] == prod_id), {})
                        p_name = product.get("Name", "General Service")
                        
                        val = table_ans.get("TextValue")
                        if val:
                            content = f"{rfx_label} | Supplier: {s_name} | Service: {p_name} | {t_q_name}: {val}"
                            documents.append(Document(
                                page_content=content, 
                                metadata={**base_meta, "supplier": s_name, "type": "product_spec"}
                            ))

    return documents


def run_live_ingestion(rfx_id):
    """Main execution loop for the live bridge."""
    try:
        # 1. Fetch
        raw_json = fetch_live_rfx_data(rfx_id)
        
        # 2. Parse
        print("Parsing JSON into searchable documents...")
        docs = parse_rfx_to_documents(raw_json)
        print(f"Created {len(docs)} context-aware chunks.")

        # 3. Embed and Store
        print("Updating Vector Database...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Use .from_documents to overwrite/create the DB
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print(f"Success! RFX {rfx_id} is now in the Brain's memory.")
        
    except Exception as e:
        print(f"Ingestion failed: {e}")

def ingest_rfxs(rfx_ids):
    """Ingests multiple RFXs based on the provided list of IDs."""
    for rfx_id in rfx_ids:
        print(f"\n--- Ingesting RFX: {rfx_id} ---")
        run_live_ingestion(rfx_id)

if __name__ == "__main__":
    parser = ArgumentParser(description="Ingest RFX data from RFX API into the vector database.")
    parser.add_argument('--ids', nargs='*', help="List of RFX IDs to ingest. Leave blank to ingest all RFXs.", default=[])
    
    args = parser.parse_args()
    
    if args.ids:
        # Ingest specific RFXs
        ingest_rfxs(args.ids)
    else:
        # Ingest all RFXs (bulk ingestion)
        # For bulk ingestion, you might want to implement a separate function or logic
        print("Ingesting all RFXs... (logic not implemented)")
        # Example: ingest_all_rfxs()