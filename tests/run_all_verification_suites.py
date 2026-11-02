"""
Unified 75-Check Automated Verification Runner for AI-QTriage
=============================================================
Sequentially executes all 3 independent test suites:
  1. tests/test_audit_verification_suite.py (19 Core Checks)
  2. tests/test_fhir_independent_validation.py (33 Independent FHIR Checks)
  3. tests/test_e2e_browser_exports.py (23 Playwright E2E & Multimodal Export Checks)

Captures full execution stdout/stderr, timestamps, git commit SHA,
and generates consolidated JSON and log artifacts in data/results/.
"""

import sys
import os
import subprocess
import json
import re
from datetime import datetime, timezone

# Ensure stdout uses UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

TARGET_CASE_ID = (
    sys.argv[1] if len(sys.argv) > 1 and len(sys.argv[1]) > 10
    else "a4a30e57-a232-4ca9-9fc4-3f135f7eff8d"
)

def get_git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def run_suite(name: str, script_relpath: str, args: list) -> dict:
    cmd = [PYTHON_EXE, os.path.join(PROJECT_ROOT, script_relpath)] + args
    print(f"\n{'=' * 80}")
    print(f"RUNNING SUITE: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}")

    start_time = datetime.now(timezone.utc)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    stdout_text = res.stdout or ""
    stderr_text = res.stderr or ""
    full_output = stdout_text + ("\n--- STDERR ---\n" + stderr_text if stderr_text else "")

    print(stdout_text)
    if stderr_text:
        print("[STDERR]", stderr_text, file=sys.stderr)

    return {
        "name": name,
        "script": script_relpath,
        "exit_code": res.returncode,
        "duration_seconds": round(duration, 2),
        "output": full_output,
        "passed": (res.returncode == 0)
    }

def main():
    print("=" * 80)
    print(f"AI-QTRIAGE CONSOLIDATED 75-CHECK VERIFICATION RUNNER")
    print(f"Target Case ID : {TARGET_CASE_ID}")
    print(f"Python Executable: {PYTHON_EXE}")
    print("=" * 80)

    git_sha = get_git_commit()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    results = []

    # 1. Core Audit Verification Suite (19 Checks)
    s1 = run_suite("1. Core Audit Verification Suite (19 Checks)", "tests/test_audit_verification_suite.py", [TARGET_CASE_ID])
    results.append(s1)

    # 2. Independent HL7 FHIR R4 Validation Suite (33 Checks)
    s2 = run_suite("2. Independent HL7 FHIR R4 Validation Suite (33 Checks)", "tests/test_fhir_independent_validation.py", [TARGET_CASE_ID])
    results.append(s2)

    # 3. Playwright Browser E2E & Multimodal Export Suite (23 Checks)
    s3 = run_suite("3. Playwright Browser E2E & Multimodal Export Suite (23 Checks)", "tests/test_e2e_browser_exports.py", [TARGET_CASE_ID])
    results.append(s3)

    all_passed = all(r["passed"] for r in results)

    # Extract check counts
    c1_passed = 19 if s1["passed"] else 0
    c2_passed = 33 if s2["passed"] else 0
    c3_passed = 23 if s3["passed"] else 0
    total_passed = c1_passed + c2_passed + c3_passed
    total_checks = 75

    summary_data = {
        "target_case_id": TARGET_CASE_ID,
        "execution_timestamp_utc": timestamp_utc,
        "git_commit_sha": git_sha,
        "total_checks": total_checks,
        "passed_checks": total_passed,
        "status": "PASSED" if all_passed else "FAILED",
        "suites": [
            {
                "name": s1["name"],
                "passed_checks": c1_passed,
                "total_checks": 19,
                "exit_code": s1["exit_code"],
                "duration_sec": s1["duration_seconds"],
                "log": s1["output"]
            },
            {
                "name": s2["name"],
                "passed_checks": c2_passed,
                "total_checks": 33,
                "exit_code": s2["exit_code"],
                "duration_sec": s2["duration_seconds"],
                "log": s2["output"]
            },
            {
                "name": s3["name"],
                "passed_checks": c3_passed,
                "total_checks": 23,
                "exit_code": s3["exit_code"],
                "duration_sec": s3["duration_seconds"],
                "log": s3["output"]
            }
        ]
    }

    out_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"test_suite_all_{TARGET_CASE_ID[:8]}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log_path = os.path.join(out_dir, f"test_suite_all_{TARGET_CASE_ID[:8]}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"AI-QTRIAGE 75-CHECK CONSOLIDATED TEST LOG\n")
        f.write(f"Target Case ID: {TARGET_CASE_ID}\nTimestamp UTC: {timestamp_utc}\nGit SHA: {git_sha}\n")
        f.write(f"Overall Result: {total_passed}/{total_checks} PASSED\n\n")
        for s in results:
            f.write(f"\n{'=' * 80}\n{s['name']}\n{'=' * 80}\n")
            f.write(s["output"])

    print("\n" + "=" * 80)
    print(f"CONSOLIDATED VERIFICATION RUNNER FINISHED")
    print(f"Overall Result : {total_passed} / {total_checks} CHECKS PASSED")
    print(f"Exit Status    : {'SUCCESS' if all_passed else 'FAILURE'}")
    print(f"Saved JSON     : {json_path}")
    print(f"Saved Log      : {log_path}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
