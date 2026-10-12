import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if 2640 <= i <= 2650:
            d = json.loads(line)
            print(f"=== Line {i}, Step {d.get('step_index')}, Type {d.get('type')} ===")
            content = d.get('content', '')
            print(content[:2000])
            print("="*60)
