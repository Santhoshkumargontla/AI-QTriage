import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            content = data.get("content", "")
            if "PAGE " in content or "Reviewer" in content or "reviewer" in content or "comment" in content:
                print(f"=== USER_INPUT at line {i}, step {data.get('step_index')} ===")
                print(content)
                print("="*60)
