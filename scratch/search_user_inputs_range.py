import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            content = data.get("content", "")
            if len(content) > 200:
                print(f"=== Line {i}, Step {data.get('step_index')} (len={len(content)}) ===")
                lines = content.strip().split("\n")
                for l in lines[:10]:
                    print("  ", l[:120])
                if len(lines) > 10:
                    print("   ... [", len(lines)-10, "more lines ]")
                    for l in lines[-5:]:
                        print("  ", l[:120])
                print("="*60)
