"""Quick local test — calls query_rfx directly, no queue needed."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

from src.retriever import query_rfx

# Test questions from the original test matrix
tests = [
    ("2100177", "Which suppliers have responded to this RFX?"),
    ("2100177", "List all product prices from every supplier."),
    ("2100149", "Which suppliers have not responded?"),
]

for rfx_id, question in tests:
    print(f"\n{'='*80}")
    print(f"RFX: {rfx_id} | Q: {question}")
    print('='*80)
    answer = query_rfx(question, rfx_id, role="default")
    print(answer)
