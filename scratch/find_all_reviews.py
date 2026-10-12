import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            content = data.get('content', '')
            if any(term in content for term in ['PAGE', 'Reviewer', 'reviewer', 'Comment', 'comment', 'Table 1', 'Table-1']):
                sub = content[:150].replace('\n', ' ')
                step = data.get('step_index')
                print(f"Line {i}, Step {step}: {sub}")
