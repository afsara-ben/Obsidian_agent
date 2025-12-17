from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from adk_agents import build_planner, build_supervisor, plan_with_adk, supervise_with_adk
from adk_screen_classifier import build_screen_classifier, classify_screen
from agents import Executor
from agents.supervisor import Supervisor as FallbackSupervisor

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

def load_test_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected test case format in {path}")


def editor_fill_plan(title: str, body: str) -> List[Dict[str, Any]]:
    """
    Minimal deterministic plan to overwrite title and body in the current editor.
    Uses select-all via keycombination (Ctrl+A) then backspace to clear.
    """
    return [
        {"action": "tap", "detail": {"x_norm": 0.50, "y_norm": 0.10}, "comment": "Focus title field"},
        {"action": "wait", "detail": {"seconds": 0.2}},
        {"action": "keycombination", "detail": {"combo_code": 11329}, "comment": "Select all (Ctrl+A, Android 13+)"},
        {"action": "keyevent", "detail": {"key_code": 67}, "comment": "Backspace to clear title"},
        {"action": "wait", "detail": {"seconds": 0.2}},
        {"action": "input_text", "detail": {"text": title}, "comment": "Type title"},
        {"action": "wait", "detail": {"seconds": 0.3}},
        {"action": "tap", "detail": {"x_norm": 0.23, "y_norm": 0.35}, "comment": "Focus body area"},
        {"action": "wait", "detail": {"seconds": 0.2}},
        {"action": "keycombination", "detail": {"combo_code": 11329}, "comment": "Select all body"},
        {"action": "keyevent", "detail": {"key_code": 67}, "comment": "Backspace to clear body"},
        {"action": "wait", "detail": {"seconds": 0.2}},
        {"action": "input_text", "detail": {"text": body}, "comment": "Body text"},
    ]


def detect_state(adb_client, dump_path: Path, screenshot_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Heuristic state detection using uiautomator dump.
    Returns {"state": str, "dump_path": str} where state is one of:
    editor | vault_config | continue_prompt | vault_splash | unknown | other_app
    Caller is responsible for taking the screenshot; we only attach its path.
    """
    print(f"detect_state: {dump_path} {screenshot_path}")
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    res = adb_client.dump_ui(dump_path)
    if res.get("status") != "ok":
        return {"state": "unknown", "dump_path": None, "error": res.get("stderr"), "screenshot_path": str(screenshot_path) if screenshot_path else None}

    try:
        tree = ET.parse(dump_path)
    except Exception:  # noqa: BLE001
        return {
            "state": "unknown",
            "dump_path": str(dump_path),
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }

    text_blobs = []
    packages = set()
    for node in tree.iter():
        pkg = node.attrib.get("package")
        if pkg:
            packages.add(pkg)
        t = (node.attrib.get("text") or "") + " " + (node.attrib.get("content-desc") or "")
        if t.strip():
            text_blobs.append(t.lower())
    full_text = " ".join(text_blobs)
    primary_pkg = packages.pop() if packages else None

    if primary_pkg and primary_pkg != "md.obsidian":
        return {
            "state": "other_app",
            "dump_path": str(dump_path),
            "package": primary_pkg,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }

    if "continue without sync" in full_text:
        return {
            "state": "continue_prompt",
            "dump_path": str(dump_path),
            "package": primary_pkg,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }
    if "configure your new vault" in full_text or "vault name" in full_text:
        return {
            "state": "vault_config",
            "dump_path": str(dump_path),
            "package": primary_pkg,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }
    if "create a vault" in full_text and "existing vault" in full_text:
        print(f"vault_splash: {full_text}")
        return {
            "state": "vault_splash",
            "dump_path": str(dump_path),
            "package": primary_pkg,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }
    if re.search(r"untitled", full_text) or "vault location" not in full_text:
        # Editor heuristic: presence of "untitled" title and absence of vault config markers.
        return {
            "state": "editor",
            "dump_path": str(dump_path),
            "package": primary_pkg,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }

    return {
        "state": "unknown",
        "dump_path": str(dump_path),
        "package": primary_pkg,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
    }


def capture_observation(adb_client, step_idx: int) -> Dict[str, Any]:
    """
    Capture screenshot and ui dump for the current step. Returns observation dict.
    """
    ss_path = Path(f"artifacts/obs_{step_idx}.png")
    dump_path = Path("artifacts/state_uidump.xml")
    ss_path.parent.mkdir(parents=True, exist_ok=True)
    adb_client.screenshot(ss_path)
    state_info = detect_state(adb_client, dump_path=dump_path, screenshot_path=ss_path)
    state_info["screenshot_path"] = str(ss_path)
    state_info["dump_path"] = str(dump_path)
    
    # Extract clickable elements from UI dump for planner to see
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(dump_path)
        clickable_elements = []
        for node in tree.iter():
            if node.attrib.get("clickable") == "true" or node.attrib.get("text"):
                text = node.attrib.get("text", "")
                bounds = node.attrib.get("bounds", "")
                resource_id = node.attrib.get("resource-id", "")
                if text or resource_id:
                    clickable_elements.append({
                        "text": text,
                        "bounds": bounds,
                        "resource_id": resource_id
                    })
        if clickable_elements:
            state_info["clickable_elements"] = clickable_elements[:10]  # Top 10 to avoid huge prompts
    except Exception as e:
        print(f"capture_observation: failed to parse UI dump: {e}")
    
    print(f"capture_observation: {state_info}")
    return state_info


def resolve_test_cases_from_prompt(prompt: Optional[str], cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If a free-form prompt is provided, attempt to route to a matching test case.
    Fallback: synthesize a single ad-hoc test case using the prompt as description.
    """
    if not prompt:
        return cases

    prompt_lower = prompt.lower()
    for case in cases:
        if case.get("id", "").lower() in prompt_lower:
            return [case]
        if case.get("description", "").lower() in prompt_lower:
            return [case]

    return [
        {
            "id": "PROMPT",
            "description": prompt,
            "expected_outcome": "pass",
            "notes": "Generated from --prompt input; ADK planner will derive steps.",
        }
    ]


def sanitize_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only executor-supported actions. Unknown actions are dropped to avoid failures.
    """
    supported = {
        "start_app",
        "stop_app",
        "clear_app",
        "tap",
        "tap_text",
        "swipe",
        "input_text",
        "keyevent",
        "keycombination",
        "wait",
        "screenshot",
        "dump_ui",
    }
    cleaned: List[Dict[str, Any]] = []
    for step in plan or []:
        action = step.get("action")
        if action in supported:
            cleaned.append(step)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mobile QA agent suite with Google ADK planner/supervisor.")
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
        default="adb",
        help="dry: simulate actions; adb: send real ADB commands.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/moondream",
        help="Model name for ADK LlmAgent planner/supervisor (default: Moondream, fast lightweight vision model).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Natural language request to choose or synthesize a test case.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum ADK planning/visual steps per test case.",
    )
    args = parser.parse_args()

    cases = resolve_test_cases_from_prompt(args.prompt, load_test_cases(args.tests))

    # ADK agents
    planner_agent = build_planner(model=args.model)
    screen_classifier = build_screen_classifier(model=args.model)
    supervisor_agent = build_supervisor(model=args.model)

    # Executors and supervisor fallback.
    executor = Executor(device_id=args.device_id, mode=args.mode)
    fallback_supervisor = FallbackSupervisor()

    results: List[Dict[str, Any]] = []

    for case in cases:
        history: List[Dict[str, Any]] = []
        execution_steps: List[Dict[str, Any]] = []
        notes: List[str] = []
        status = "ok"
        last_observation: Dict[str, Any] = {"state": "bootstrap"}

        # Always bootstrap the app so the first visual observation is meaningful.
        bootstrap_plan = [
            {
                "action": "start_app",
                "detail": {"package": "md.obsidian", "activity": ".MainActivity"},
                "comment": "Bootstrap: ensure Obsidian is launched.",
            },
            {"action": "wait", "detail": {"seconds": 3}, "comment": "Give the app time to render."},
        ]
        bootstrap_exec = executor.execute(bootstrap_plan, case, last_observation)
        execution_steps.extend(bootstrap_exec.get("steps", []))
        notes.extend(bootstrap_exec.get("notes", []))
        if bootstrap_exec.get("status") != "ok":
            status = "fail"
        history.append({"observation": last_observation, "plan": bootstrap_plan, "execution": bootstrap_exec})

        for step_idx in range(args.max_steps):
            observation = capture_observation(executor.adb, step_idx)
            observation["visual_state"] = classify_screen(screen_classifier, Path(observation["screenshot_path"]))
            # If we confidently classify the screen, propagate it to state for planning.
            if observation.get("visual_state") and observation["visual_state"] != "unknown":
                observation["state"] = observation["visual_state"]
            print(f"visual_state: {observation['visual_state']}")
            print(
                f"[obs {step_idx}] state={observation.get('state')} visual={observation.get('visual_state')} "
                f"screenshot={observation.get('screenshot_path')} dump={observation.get('dump_path')}"
            )
            last_observation = observation

            # Ask the ADK planner for the next small set of steps, using visual context.
            plan_raw = plan_with_adk(planner_agent, case, observation, history)
            plan = sanitize_plan(plan_raw)
            print(f"[planner {step_idx}] raw={plan_raw} sanitized={plan}")

            # If the planner produced nothing actionable, stop to avoid rule-based fallbacks.
            if not plan:
                status = "fail"
                notes.append("Planner returned no actionable steps; stopping to avoid rule-based fallback.")
                print(f"[planner {step_idx}] empty after sanitize; raw={plan_raw}")
                break

            exec_result = executor.execute(plan, case, observation)
            execution_steps.extend(exec_result.get("steps", []))
            notes.extend(exec_result.get("notes", []))
            history.append({"observation": observation, "plan": plan, "execution": exec_result})

            if exec_result.get("status") != "ok":
                status = "fail"
                break

            # Stop early if we've reached the editor screen; vault creation should be done.
            if observation.get("visual_state") == "editor":
                break

        execution = {"status": status, "steps": execution_steps, "notes": notes, "observation": last_observation}

        # Debug trace of steps with resolved coordinates if present.
        steps = execution.get("steps", [])
        if steps:
            print("Step trace:")
            for s in steps:
                resolved = ""
                if "resolved_x" in s.get("detail", {}):
                    resolved = f" (x={s['detail'].get('resolved_x')}, y={s['detail'].get('resolved_y')})"
                elif "resolved_start_x" in s.get("detail", {}):
                    resolved = (
                        f" (start=({s['detail'].get('resolved_start_x')},{s['detail'].get('resolved_start_y')}), "
                        f"end=({s['detail'].get('resolved_end_x')},{s['detail'].get('resolved_end_y')}))"
                    )
                print(
                    f"  step {s.get('step_index')}: {s.get('action')} -> {s.get('result')}{resolved}"
                )
                stderr = s.get("stderr")
                if stderr:
                    print(f"    stderr: {stderr}")

        # Try ADK supervisor; fall back to rule-based if parsing fails.
        verdict = supervise_with_adk(supervisor_agent, case, execution)
        if not verdict or "verdict" not in verdict:
            verdict = fallback_supervisor.evaluate(case, execution)

        result = {
            "test_id": case.get("id"),
            "description": case.get("description"),
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "expected_outcome": verdict.get("expected_outcome"),
        }
        results.append(result)
        print(f"[{result['test_id']}] {result['verdict']} - {result['reason']}")

    passed = sum(1 for r in results if r["verdict"] == "pass")
    print(f"Completed {len(results)} tests: {passed} passed, {len(results) - passed} failed.")


if __name__ == "__main__":
    main()

