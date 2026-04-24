"""
LoanXAI — API Test Script
Loads 3 test cases from sample_data/sample_requests.json
Tests each against POST /predict
Validates response structure and values
"""

import json
import sys
import os
import requests

API_BASE = "http://localhost:5000"
SAMPLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_data", "sample_requests.json")

VALID_DECISIONS = {"APPROVED", "REJECTED", "MANUAL REVIEW"}
VALID_SOURCES   = {"ML_MODEL", "RULE_BASED"}


def run_tests():
    # Load test cases
    with open(SAMPLE_FILE, "r") as f:
        test_cases = json.load(f)

    print("=" * 56)
    print("  LoanXAI API Test Suite")
    print("=" * 56)
    print(f"\nLoaded {len(test_cases)} test cases from sample_requests.json\n")

    # Check health first
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        h = health.json()
        print(f"[HEALTH] status={h['status']}  model={h['model']}  features={h['features']}")
        print()
    except Exception as e:
        print(f"[FATAL] Cannot reach backend at {API_BASE}/health")
        print(f"        Error: {e}")
        print(f"        Make sure Flask is running: python backend/app.py")
        sys.exit(1)

    passed = 0
    total = len(test_cases)

    for i, tc in enumerate(test_cases):
        name = tc.get("name", f"Test Case {i+1}")
        comment = tc.get("_comment", "")
        print(f"--- Test {i+1}: {name} ---")
        if comment:
            print(f"    Scenario: {comment}")

        errors = []

        try:
            res = requests.post(f"{API_BASE}/predict", json=tc, timeout=15)
            data = res.json()

            if "error" in data:
                errors.append(f"API returned error: {data['error']}")
            else:
                # Assert 1: decision is valid
                decision = data.get("decision")
                if decision not in VALID_DECISIONS:
                    errors.append(f"Invalid decision: '{decision}' (expected one of {VALID_DECISIONS})")

                # Assert 2: source is valid
                source = data.get("source", "UNKNOWN")
                if source not in VALID_SOURCES:
                    errors.append(f"Invalid source: '{source}' (expected one of {VALID_SOURCES})")

                # Assert 3: risk_score is between 0 and 100
                risk = data.get("risk_score")
                if risk is None or not (0 <= risk <= 100):
                    errors.append(f"risk_score out of range: {risk} (expected 0-100)")

                # Assert 4: approval_probability + default_probability = 100
                approval = data.get("approval_probability", 0)
                default = data.get("default_probability", 0)
                total_prob = round(approval + default, 1)
                if total_prob != 100.0:
                    errors.append(f"approval_probability ({approval}) + default_probability ({default}) = {total_prob}, expected 100.0")

                # Assert 5: SHAP factors (skip for RULE_BASED)
                factors = data.get("factors", [])
                if source == "ML_MODEL":
                    if len(factors) < 5:
                        errors.append(f"Only {len(factors)} SHAP factors returned (expected >= 5 for ML_MODEL)")
                    # Assert 5b: each factor has required fields
                    required_fields = {"feature", "label", "direction", "explanation"}
                    for j, factor in enumerate(factors):
                        missing = required_fields - set(factor.keys())
                        if missing:
                            errors.append(f"Factor {j+1} missing fields: {missing}")
                            break
                elif source == "RULE_BASED":
                    if len(factors) != 0:
                        errors.append(f"RULE_BASED should have 0 factors, got {len(factors)}")
                    if not data.get("rule_reason"):
                        errors.append("RULE_BASED response missing rule_reason")

                # Assert 6: actions list has at least 1 item
                actions = data.get("actions", [])
                if len(actions) < 1:
                    errors.append("No action items returned (expected >= 1)")

                # Assert 7: confidence level exists
                confidence = data.get("confidence", {})
                if not confidence.get("level"):
                    errors.append("Missing confidence level")

                # Print summary
                print(f"    Decision:    {decision}")
                print(f"    Source:      {source}")
                print(f"    Risk Score:  {risk}")
                print(f"    Approval:    {approval}%")
                print(f"    Confidence:  {confidence.get('level', 'N/A')}")
                print(f"    Factors:     {len(factors)}")
                print(f"    Actions:     {len(actions)}")
                if source == "RULE_BASED":
                    print(f"    Rule:        {data.get('rule', 'N/A')}")
                if data.get('calibration_note'):
                    print(f"    Calibration: {data['calibration_note']}")

        except requests.exceptions.ConnectionError:
            errors.append("Connection refused - is the Flask server running?")
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")

        if errors:
            print(f"    Result:      FAIL")
            for err in errors:
                print(f"    [ERROR] {err}")
        else:
            print(f"    Result:      PASS")
            passed += 1

        print()

    # Summary
    print("=" * 56)
    if passed == total:
        print(f"  PASSED {passed}/{total} tests")
    else:
        print(f"  PASSED {passed}/{total} tests  ({total - passed} FAILED)")
    print("=" * 56)

    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
