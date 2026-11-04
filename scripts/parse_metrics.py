import json
import os

log_path = os.path.join("outputs", "unlimited-ocr2-indic-lora", "training_progress.jsonl")
records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

print(f"Total log entries: {len(records)}")
# Print summary of key milestones
for idx in [0, 10, 50, 100, 113, 150, 200, len(records)-1]:
    if idx < len(records):
        r = records[idx]
        print(f"Index {idx:3d} | Step {r.get('step', -1):3d} | Epoch {r.get('epoch', 0):.2f} | Loss: {r.get('loss', 0):.4f} | LR: {r.get('learning_rate', 0):.2e}")
