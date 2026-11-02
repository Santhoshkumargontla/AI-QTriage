import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("CHECKLIST 6: RESEARCH METRICS, DATA SPLIT & SCIENTIFIC TRANSPARENCY AUDIT")
print("=" * 80)

results_dir = os.path.join(PROJECT_ROOT, "data", "results")

# 1. Load canonical held-out evaluation
held_out_path = os.path.join(results_dir, "canonical_held_out_evaluation.json")
with open(held_out_path, "r", encoding="utf-8") as f:
    held_out = json.load(f)

# 2. Load VQC vs XGBoost comparison
compare_path = os.path.join(results_dir, "vqc_xgb_comparison.json")
with open(compare_path, "r", encoding="utf-8") as f:
    comparison = json.load(f)

# 3. Load duplicate split audit
split_audit_path = os.path.join(results_dir, "phase1_duplicate_split_audit.json")
with open(split_audit_path, "r", encoding="utf-8") as f:
    split_audit = json.load(f)

checks = []

# Metric verification
xgb_acc = held_out["comparison"]["xgboost"]["accuracy"]
vqc_acc = held_out["comparison"]["vqc"]["accuracy"]
checks.append(("XGBoost held-out accuracy is 83.33% (25/30)", abs(xgb_acc - 0.833333) < 1e-4, f"XGB={xgb_acc*100:.2f}%"))
checks.append(("VQC held-out accuracy is 56.67% (17/30)", abs(vqc_acc - 0.566667) < 1e-4, f"VQC={vqc_acc*100:.2f}%"))
checks.append(("Classical beats simulated Quantum VQC", xgb_acc > vqc_acc, f"Delta: +{(xgb_acc - vqc_acc)*100:.2f}% for XGBoost"))

# Quantum claim audit
checks.append(("Quantum outperforming XGB: FALSE", comparison.get("vqc_outperforms_xgboost") is False, "VQC does NOT outperform XGBoost"))
checks.append(("VQC used in main decision: FALSE", comparison.get("used_in_main_decision") is False, "Recommendation: DISABLE_FROM_MAIN_DECISION"))
checks.append(("Simulated Qubits count: 4", comparison.get("circuit_match", {}).get("num_qubits") == 4, "4 qubits classical simulation on default.qubit"))

# Zero-leakage & Data Split audit
overlap = held_out["comparison"]["leakage_audit"]["exact_row_hash_overlap"]
checks.append(("Exact-row hash overlap train-test is 0", overlap.get("train_test") == 0, "0 overlap rows"))
checks.append(("Exact-row hash overlap train-val is 0", overlap.get("train_val") == 0, "0 overlap rows"))
checks.append(("Subject-level limitation explicitly disclosed", "SUBJECT_LEAKAGE_NOT_VERIFIABLE" in held_out["comparison"]["leakage_audit"]["subjects"], "Disclosed: synthetic vectors have no subject_id"))
checks.append(("Data provenance class is SYNTHETIC", comparison.get("data_provenance_class") == "SYNTHETIC", "SYNTHETIC"))

all_passed = True
for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status}] {name:46s}: {detail}")

print("=" * 80)
if all_passed:
    print(f"ALL {len(checks)} RESEARCH METRICS AUDIT CHECKS PASSED.")
else:
    print("SOME AUDIT CHECKS FAILED.")
    sys.exit(1)
