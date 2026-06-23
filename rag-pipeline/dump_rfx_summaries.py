"""Dump all RFX data summaries to text files for manual Q&A generation."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_ground_truth import fetch_all_rfx_ids, fetch_rfx_data, summarize_rfx_for_context

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rfx_summaries")
os.makedirs(OUTPUT_DIR, exist_ok=True)

rfx_ids = fetch_all_rfx_ids()
print(f"Found {len(rfx_ids)} RFXs")

for i, rfx_id in enumerate(rfx_ids, 1):
    print(f"[{i}/{len(rfx_ids)}] Fetching RFX {rfx_id}...")
    try:
        data = fetch_rfx_data(rfx_id)
        rfx_name = data.get("name", f"RFX_{rfx_id}")
        summary = summarize_rfx_for_context(data)

        # Save summary
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in rfx_name).strip()
        filename = f"rfx_{rfx_id}_{safe_name[:50]}.txt"
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(summary)

        # Also save raw JSON for reference
        raw_filename = f"rfx_{rfx_id}_raw.json"
        with open(os.path.join(OUTPUT_DIR, raw_filename), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  -> {rfx_name} ({len(summary)} chars)")
    except Exception as e:
        print(f"  -> FAILED: {e}")
    if i < len(rfx_ids):
        time.sleep(1)

print("Done!")
