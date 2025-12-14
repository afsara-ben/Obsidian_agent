from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from agents import Executor, Planner, Supervisor


def load_test_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected test case format in {path}")


class TestRunner:
    def __init__(self, device_id: str | None, mode: str) -> None:
        self.planner = Planner()
        self.executor = Executor(device_id=device_id, mode=mode)
        self.supervisor = Supervisor()

    def run_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        observation = {"state": "start"}
        history: List[Dict[str, Any]] = []

        plan = self.planner.plan(test_case, observation, history)
        execution = self.executor.execute(plan, test_case, observation)
        verdict = self.supervisor.evaluate(test_case, execution)

        return {
            "test_id": test_case.get("id"),
            "description": test_case.get("description"),
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "expected_outcome": verdict.get("expected_outcome"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mobile QA agent suite.")
    parser.add_argument("--tests", type=Path, default=Path("tests/test_cases.json"), help="Path to test cases JSON file.")
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="ADB device/emulator id. Leave empty for default emulator.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry", "adb"],
        default="dry",
        help="dry: simulate actions; adb: send real ADB commands.",
    )
    args = parser.parse_args()

    cases = load_test_cases(args.tests)
    runner = TestRunner(device_id=args.device_id, mode=args.mode)

    results: List[Dict[str, Any]] = []
    for case in cases:
        result = runner.run_case(case)
        results.append(result)
        print(f"[{result['test_id']}] {result['verdict']} - {result['reason']}")

    passed = sum(1 for r in results if r["verdict"] == "pass")
    print(f"Completed {len(results)} tests: {passed} passed, {len(results) - passed} failed.")


if __name__ == "__main__":
    main()

