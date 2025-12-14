from __future__ import annotations

from typing import Any, Dict


class Supervisor:
    """
    Compares executor results to expected outcomes and classifies pass/fail.
    Distinguishes between action-level failures and assertion mismatches.
    """

    def evaluate(self, test_case: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        expected_outcome = test_case.get("expected_outcome", "pass")
        status = execution.get("status", "fail")
        notes = execution.get("notes", []) or []

        if status == "fail" and expected_outcome == "fail":
            verdict = "pass"
            reason = "Expected failure observed; assertion mismatch confirmed."
        elif status == "fail" and expected_outcome != "fail":
            verdict = "fail"
            reason = "Execution failed unexpectedly."
        elif status == "ok" and expected_outcome == "fail":
            verdict = "fail"
            reason = "Test was expected to fail, but no failure was detected."
        else:
            verdict = "pass"
            reason = "Execution completed without blocking issues."

        return {
            "test_id": test_case.get("id"),
            "verdict": verdict,
            "reason": reason,
            "expected_outcome": expected_outcome,
            "execution_notes": notes,
            "execution_steps": execution.get("steps", []),
        }

