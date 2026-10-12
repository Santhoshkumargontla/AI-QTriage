import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if 2635 <= i <= 2650:
            data = json.loads(line)
            content = data.get("content", "")
            if content:
                print(f"=== Line {i}, Step {data.get('step_index')}, Type {data.get('type')} ===")
                print(content)
                print("\n" + "="*60 + "\n")
