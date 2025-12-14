from __future__ import annotations

from typing import Any, Dict, List


class Planner:
    """Turns natural language test cases into a rough action plan."""

    def plan(self, test_case: Dict[str, Any], observation: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Produce a minimal plan based on the current test description.

        This basic version keeps the plan simple: it echoes the test goal so the
        executor can decide how to act. A richer implementation would break this
        into UI-level steps based on screenshots and previous attempts.
        """
        # If a scripted plan is provided, use it directly. This keeps the
        # planner deterministic for the initial baseline and allows tweaking
        # coordinates without touching code.
        script = test_case.get("script")
        if isinstance(script, list):
            return script

        description = test_case.get("description", "")
        return [
            {
                "action": "interpret_goal",
                "detail": description,
                "commentary": "Baseline plan: describe desired end state for executor.",
            }
        ]

