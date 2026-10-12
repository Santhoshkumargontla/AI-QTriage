import json
import os

path = r"C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript.jsonl"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                content = str(data.get("content", ""))
                thinking = str(data.get("thinking", ""))
                combined = content + " " + thinking
                if "reviewer" in combined.lower() or "11 comment" in combined.lower():
                    print(f"Line {line_num}, step {data.get('step_index')}, type {data.get('type')}")
                    idx = combined.lower().find("reviewer")
                    if idx != -1:
                        print(combined[max(0, idx-50):min(len(combined), idx+250)])
                    else:
                        idx2 = combined.lower().find("11 comment")
                        print(combined[max(0, idx2-50):min(len(combined), idx2+250)])
                    print("="*50)
            except Exception as e:
                pass
