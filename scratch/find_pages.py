import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        content = data.get("content", "")
        # look for page review comments
        if "PAGE 1" in content or "PAGE 2" in content or "PAGE 4" in content or "PAGE 6" in content or "PAGE 7" in content or "PAGE 8" in content:
            print(f"Match at line {i}, step {data.get('step_index')}, type {data.get('type')}")
            for l in content.split("\n"):
                if "PAGE" in l or "Remaining" in l or "Reviewer" in l:
                    print("  ", l[:120])
            print("="*60)
