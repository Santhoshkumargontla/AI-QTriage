import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:\Users\santh\.gemini\antigravity\brain\1869e667-746c-4ef1-8933-dc346fe33226\.system_generated\logs\transcript_full.jsonl'
lines_to_check = [2559, 2617, 2642, 2668, 2671, 2677, 2693, 2695, 2697, 2701, 2703, 2707, 2709, 2711, 2720, 2722]
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in lines_to_check:
            data = json.loads(line)
            txt = data.get('content', '')
            step = data.get('step_index')
            print(f"=== LINE {i} (Step {step}) ===")
            print(txt[:300].replace('\n', ' '))
