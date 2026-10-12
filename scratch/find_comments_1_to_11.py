import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        txt = data.get('content', '')
        if any(term in txt for term in ['Reviewer 1', 'Reviewer 2', 'Reviewer #1', 'Reviewer #2', '11 reviewer comments', 'Comment 11', '11 comments']):
            print(f"Line {i}, Step {data.get('step_index')}, Type {data.get('type')}")
            lines = txt.split('\n')
            for l in lines:
                if any(k in l.lower() for k in ['reviewer', 'comment 1', 'comment 2', 'comment 3', 'comment 4', 'comment 5', 'comment 6', 'comment 7', 'comment 8', 'comment 9', 'comment 10', 'comment 11']):
                    print("  ", l[:120])
