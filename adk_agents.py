from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from google.adk.agents import LlmAgent
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "google-adk is required. Install with `pip install google-adk`."
    ) from exc


def build_planner(model: str = "gemini-2.5-flash") -> LlmAgent:
    """
    Planner agent: turns the test case + observation into a minimal action plan.
    Returns a JSON array of steps: [{action, detail, comment}]
    """
    return LlmAgent(
        name="planner",
        model=model,
        instruction=(
            "You are a mobile QA planner. Given a test case description, the last observation, "
            "and prior history, return the NEXT small set of actions to attempt. "
            "Return JSON list of steps, each with fields: action, detail, comment. "
            "Allowed actions: start_app, tap, swipe, input_text, keyevent, wait, screenshot, dump_ui. "
            "Use coordinates in normalized form when possible (x_norm,y_norm in [0,1]). "
            "Limit to 3 steps per reply."
        ),
    )


def build_supervisor(model: str = "gemini-2.5-flash") -> LlmAgent:
    """
    Supervisor agent: classifies pass/fail vs expected_outcome based on execution trace.
    """
    return LlmAgent(
        name="supervisor",
        model=model,
        instruction=(
            "You are a QA supervisor. Given expected_outcome and execution trace, "
            "decide verdict pass/fail. If expected_outcome=='fail' and execution hit a failure, "
            "verdict is pass (expected fail observed). Otherwise pass iff status is ok. "
            "Return a short reason."
        ),
    )


def plan_with_adk(planner: LlmAgent, test_case: Dict[str, Any], observation: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Invoke the ADK planner to get a list of steps. Falls back to a single interpret action if parsing fails.
    """
    prompt_payload: Dict[str, Any] = {
        "test_case": test_case,
        "observation": observation,
        "history": history[-5:],  # keep prompt small
    }
    # Encourage vision reasoning if a screenshot is available.
    if observation.get("screenshot_path"):
        prompt_payload["instruction"] = (
            "A screenshot is available at observation.screenshot_path. Use it to reason about the current UI."
        )

    prompt = json.dumps(prompt_payload, ensure_ascii=False)
    try:
        raw = planner(prompt)
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict) and "output" in raw:
            text = raw["output"]
        else:
            text = str(raw)
        steps = json.loads(text)
        if isinstance(steps, list):
            return steps
    except Exception:
        # Ignore and fall through to fallback.
        pass
    return [
        {
            "action": "interpret_goal",
            "detail": test_case.get("description", ""),
            "comment": "Fallback plan from ADK planner parsing failure.",
        }
    ]


def supervise_with_adk(supervisor: LlmAgent, test_case: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke the ADK supervisor to classify verdict. Falls back to rule-based if needed.
    """
    prompt = json.dumps(
        {
            "expected_outcome": test_case.get("expected_outcome", "pass"),
            "execution": execution,
        },
        ensure_ascii=False,
    )
    try:
        raw = supervisor(prompt)
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict) and "output" in raw:
            text = raw["output"]
        else:
            text = str(raw)
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "verdict" in parsed:
            return parsed
    except Exception:
        pass

    # Fallback to existing rule-based logic.
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
        "verdict": verdict,
        "reason": reason,
        "expected_outcome": expected_outcome,
        "execution_notes": notes,
        "execution_steps": execution.get("steps", []),
    }

