"""
Independent Verification of Gemini First-Aid Bounded Latency and Thread Pool Concurrency.
Tests realistic timeout handling, concurrent queuing, controlled mock delays, and active thread bounding.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from backend.services.gemini_first_aid_service import GeminiFirstAidService, _GEMINI_EXECUTOR

def test_gemini_concurrency_and_timeout():
    print("=" * 76)
    print("GEMINI FIRST-AID CONCURRENCY & BOUNDED LATENCY BENCHMARK")
    print("=" * 76)

    service = GeminiFirstAidService()
    dummy_evidence = {
        "case_id": "test-latency-bench",
        "questionnaire": {"answers": {"pain_level": 7, "crack_pop": "yes"}},
        "sensor_summary": {"peak_g_force": 5.17},
        "safety_rules": {"rule_level": "HIGH"}
    }

    # 1. Controlled Mock Delay & Socket Timeout Assertion
    print("\n[Test 1] Testing Controlled Mock Provider Delay & Socket Timeout Fallback...")
    
    def slow_mock_generate(*args, **kwargs):
        time.sleep(3.0)  # Simulate slow network response
        return {"status": "success", "guidance": "Mock clinical advice"}

    with patch.object(service, "_get_api_key", return_value="fake-test-key-for-mock"), \
         patch.object(service, "_call_gemini_sdk", side_effect=slow_mock_generate):
        t0 = time.time()
        timeout_limit = 1.0
        result = service.generate_guidance(dummy_evidence, timeout_seconds=timeout_limit)
        elapsed = time.time() - t0

    print(f"Elapsed Time: {elapsed:.3f}s (Configured Timeout Limit: {timeout_limit}s)")
    print(f"Result Provider: {result.get('provider')}, Status: {result.get('status')}")
    print(f"Fallback Reason: {result.get('fallback_reason')}")
    assert elapsed < timeout_limit + 0.5, f"Call did not abort within timeout bound! Took {elapsed:.2f}s"
    assert result.get("status") == "fallback", f"Expected fallback, got {result.get('status')}"
    assert result.get("provider") == "rule_based_fallback", f"Attribution must be rule_based_fallback, got {result.get('provider')}"
    assert "timeout" in str(result.get("fallback_reason")).lower(), "Reason must mention timeout"
    print("[PASS] Controlled Mock Timeout: Execution aborted cleanly within 1.0s and fell back deterministically.")

    # 2. In-Flight Concurrency Peak Measurement (4 concurrent requests, max_workers=2)
    print("\n[Test 2] Testing Active In-Flight Concurrency Cap (max_workers=2)...")
    peak_concurrency = 0
    current_concurrency = 0
    concurrency_lock = threading.Lock()

    def monitored_mock_call(*args, **kwargs):
        nonlocal peak_concurrency, current_concurrency
        with concurrency_lock:
            current_concurrency += 1
            if current_concurrency > peak_concurrency:
                peak_concurrency = current_concurrency
        time.sleep(0.3)  # Hold worker slot
        with concurrency_lock:
            current_concurrency -= 1
        return {"status": "success", "guidance": "Simulated LLM advice"}

    results = []
    threads = []

    with patch.object(service, "_get_api_key", return_value="fake-test-key-for-mock"), \
         patch.object(service, "_call_gemini_sdk", side_effect=monitored_mock_call):
        t_start_all = time.time()
        for i in range(4):
            def worker(idx):
                t_w0 = time.time()
                res = service.generate_guidance(dummy_evidence, timeout_seconds=3.0)
                dt = time.time() - t_w0
                results.append((idx, res.get("status"), dt))

            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    total_time = time.time() - t_start_all
    print(f"All 4 calls completed in {total_time:.3f}s")
    for idx, status, dt in sorted(results):
        print(f"  Worker {idx}: status={status}, latency={dt:.3f}s")

    print(f"\nMeasured Peak In-Flight Concurrency: {peak_concurrency}")
    print(f"Configured ThreadPool max_workers: {_GEMINI_EXECUTOR._max_workers}")
    assert peak_concurrency <= 2, f"Concurrency exceeded pool cap! Peak was {peak_concurrency}"
    assert _GEMINI_EXECUTOR._max_workers == 2, f"ThreadPool executor max_workers is {_GEMINI_EXECUTOR._max_workers}"
    print("[PASS] In-Flight Concurrency strictly capped at 2. No thread saturation possible.")
    print("=" * 76)

if __name__ == "__main__":
    test_gemini_concurrency_and_timeout()
