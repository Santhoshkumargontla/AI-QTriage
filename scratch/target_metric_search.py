import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

targets = ['36.47', '18.45', '18.44', '83.58', '70.80', '64.18', '93.75', '1183', '1228', '96.34', '77.78', '0.822222', '0.702692', '0.118284', '0.304532', '0.095978', '0.136455', '0.512306', '0.769281', '0.785020', '0.466184', '0.432680', '0.423016']

dirs_to_check = ['ml', 'backend', 'frontend/src', 'docs', 'scratch']

for d in dirs_to_check:
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith(('.py', '.json', '.md', '.txt', '.ts', '.tsx')):
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as handle:
                        txt = handle.read()
                    matches = [t for t in targets if t in txt]
                    if matches:
                        print(f"{p}: matches {matches}", flush=True)
                except:
                    pass
