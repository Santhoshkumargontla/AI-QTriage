import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if 2550 <= i <= 2645:
            d = json.loads(line)
            if d.get("type") == "USER_INPUT":
                print(f"Line {i}: step {d.get('step_index')}")
                print(d.get("content"))
                print("="*60)
