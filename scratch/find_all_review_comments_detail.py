import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        content = data.get('content', '')
        clow = content.lower()
        if '11 reviewer' in clow or '11 comments' in clow or ('reviewer' in clow and 'comment' in clow) or 'reviewer 1' in clow or 'reviewer 2' in clow:
            step = data.get('step_index')
            m = clow.count('reviewer')
            t = data.get('type')
            print(f"Line {i}, Step {step}, Type {t}, Occurrences: {m}")
            if t == 'USER_INPUT':
                print("--- USER_INPUT snippet ---")
                print(content[:500])
                print("--------------------------")
