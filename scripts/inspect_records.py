import json

records = [json.loads(l) for l in open('outputs/unlimited-ocr2-indic-lora/training_progress.jsonl', 'r')]
print(f"Count: {len(records)}")
min_loss = min(r['loss'] for r in records)
max_loss = max(r['loss'] for r in records)
latest = records[-1]
print(f"Min loss: {min_loss:.4f}")
print(f"Max loss: {max_loss:.4f}")
print(f"Latest record: {latest}")
