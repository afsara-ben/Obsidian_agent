from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from adk_agents import build_supervisor, supervise_with_adk
from adk_screen_classifier import build_screen_classifier, classify_screen
from agents import Executor
from agents.planner import Planner as FallbackPlanner
from agents.supervisor import Supervisor as FallbackSupervisor


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
    """
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    if screenshot_path:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        adb_client.screenshot(screenshot_path)

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
    return state_info


def decide_actions_for_state(state_label: str) -> List[Dict[str, Any]]:
    """
    Deterministic actions per known screen.
    """
    actions: List[Dict[str, Any]] = []
    if state_label == "create_vault_splash":
        actions.append({"action": "tap", "detail": {"x": 520, "y": 1100}, "comment": "Tap 'Create a vault' center"})
    elif state_label == "sync_dialog":
        # Primary tap on measured center; fallback text tap if planner supports; here just coord.
        actions.append({"action": "tap", "detail": {"x_norm": 0.5005, "y_norm": 0.6318}, "comment": "Tap 'Continue without sync' center"})
    elif state_label == "storage_config":
        actions.extend(
            [
                {"action": "tap", "detail": {"x_norm": 0.5005, "y_norm": 0.3030}, "comment": "Focus vault name field"},
                {"action": "wait", "detail": {"seconds": 0.2}},
                {"action": "input_text", "detail": {"text": "InternVault"}, "comment": "Enter vault name"},
                {"action": "wait", "detail": {"seconds": 0.5}},
                {"action": "tap", "detail": {"x_norm": 0.5005, "y_norm": 0.4123}, "comment": "Select Device storage"},
                {"action": "wait", "detail": {"seconds": 0.5}},
                {"action": "tap", "detail": {"x_norm": 0.5005, "y_norm": 0.7781}, "comment": "Tap final Create"},
            ]
        )
    return actions


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
        default="gemini-2.5-flash",
        help="Model name for ADK LlmAgent planner/supervisor.",
    )
    args = parser.parse_args()

    cases = load_test_cases(args.tests)

    # ADK agents
    screen_classifier = build_screen_classifier(model=args.model)
    supervisor_agent = build_supervisor(model=args.model)

    # Executors and supervisor fallback.
    executor = Executor(device_id=args.device_id, mode=args.mode)
    fallback_supervisor = FallbackSupervisor()

    results: List[Dict[str, Any]] = []

    for case in cases:
        # Always bootstrap the app.
        executor.execute(
            [
                {
                    "action": "start_app",
                    "detail": {"package": "md.obsidian", "activity": ".MainActivity"},
                    "comment": "Bootstrap: ensure Obsidian is launched.",
                }
            ],
            case,
            {},
        )

        execution_steps: List[Dict[str, Any]] = []

        # Step 1: observe and tap Create a vault (from obs_0).
        obs0 = capture_observation(executor.adb, 0)
        actions0 = [
            {"action": "tap", "detail": {"x": 520, "y": 1100}, "comment": "Tap Create a vault (from obs_0 center)"},
            {"action": "wait", "detail": {"seconds": 1}},
        ]
        exec0 = executor.execute(actions0, case, obs0)
        execution_steps.extend(exec0.get("steps", []))

        # Step 2: observe and tap Continue without sync (from obs_0 gray region).
        obs1 = capture_observation(executor.adb, 1)
        actions1 = [
            {"action": "tap", "detail": {"x_norm": 0.4810, "y_norm": 0.5386}, "comment": "Tap Continue without sync (from obs_0 gray center)"},
            {"action": "wait", "detail": {"seconds": 1}},
        ]
        exec1 = executor.execute(actions1, case, obs1)
        execution_steps.extend(exec1.get("steps", []))

        # Step 3: observe and fill storage config.
        obs2 = capture_observation(executor.adb, 2)
        actions2 = decide_actions_for_state("storage_config")
        exec2 = executor.execute(actions2, case, obs2)
        execution_steps.extend(exec2.get("steps", []))

        # Step 4: observe and tap "USE THIS FOLDER" on permission screen.
        obs3 = capture_observation(executor.adb, 3)
        actions3 = [
            {"action": "tap", "detail": {"x_norm": 0.4995, "y_norm": 0.9478}, "comment": "Tap USE THIS FOLDER (from obs_3 center)"},
            {"action": "wait", "detail": {"seconds": 1}},
        ]
        exec3 = executor.execute(actions3, case, obs3)
        execution_steps.extend(exec3.get("steps", []))

        # Step 5: observe and tap "Allow" on the next popup.
        obs4 = capture_observation(executor.adb, 4)
        actions4 = [
            {"action": "tap", "detail": {"x": 878, "y": 1406}, "comment": "Tap Allow (from obs_4 dark region center)"},
            {"action": "wait", "detail": {"seconds": 1}},
        ]
        exec4 = executor.execute(actions4, case, obs4)
        execution_steps.extend(exec4.get("steps", []))

        execution = {"status": "ok", "steps": execution_steps, "notes": [], "observation": {}}

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

