import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        content = data.get("content", "")
        if "PAGE 3" in content or "Remaining scientific weakness" in content or "PAGE 1" in content or "PAGE 2" in content:
            print(f"Line {i}, Step {data.get('step_index')}, Type {data.get('type')}")
            print(content[:1500])
            print("="*60)
