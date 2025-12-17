from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from tools.adb_client import AdbClient

try:
    from google.adk.agents import LlmAgent
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "google-adk is required. Install with `pip install google-adk`."
    ) from exc


def build_planner(model: str = "ollama/moondream") -> LlmAgent:
    """
    Planner agent: turns the test case + observation into a minimal action plan.
    Returns a JSON array of steps: [{action, detail, comment}]
    Uses local Moondream model via Ollama by default (fast, lightweight vision model).
    """
    # Use LiteLlm wrapper for Ollama models
    if isinstance(model, str) and model.startswith("ollama/"):
        model_obj = LiteLlm(model=model)
    else:
        model_obj = model
    
    return LlmAgent(
        name="planner",
        model=model_obj,
        instruction=(
            "You are a mobile QA planner. Given a test case description, the last observation, "
            "and prior history, return the NEXT small set of actions to attempt. "
            "Return JSON list of steps, each with fields: action, detail, comment. "
            "Allowed actions: start_app, tap, swipe, input_text, keyevent, wait, screenshot, dump_ui. "
            "Use coordinates in normalized form when possible (x_norm,y_norm in [0,1]). "
            "Limit to 3 steps per reply."
        ),
    )


def build_supervisor(model: str = "ollama/moondream") -> LlmAgent:
    """
    Supervisor agent: classifies pass/fail vs expected_outcome based on execution trace.
    Uses local Moondream model via Ollama by default (fast, lightweight).
    """
    # Use LiteLlm wrapper for Ollama models
    if isinstance(model, str) and model.startswith("ollama/"):
        model_obj = LiteLlm(model=model)
    else:
        model_obj = model
    
    return LlmAgent(
        name="supervisor",
        model=model_obj,
        instruction=(
            "You are a QA supervisor. Given expected_outcome and execution trace, "
            "decide verdict pass/fail. If expected_outcome=='fail' and execution hit a failure, "
            "verdict is pass (expected fail observed). Otherwise pass iff status is ok. "
            "Return a short reason."
        ),
    )


def build_ui_reader(model: str = "ollama/moondream") -> Agent:
    """
    UI reader agent backed by a local LiteLlm model; extracts visible text/UI elements from a screenshot.
    """
    return Agent(
        name="ui_reader",
        model=LiteLlm(model=model),
        instruction="Read the UI screenshot and extract visible text and key UI elements.",
    )


def _call_ollama(model: str, prompt: str) -> Optional[str]:
    """
    Minimal local call to an Ollama-hosted model (e.g., ollama/qwen2.5vl:7b).
    Expects Ollama running at localhost:11434.
    For vision models, extracts screenshot_path from the prompt JSON and sends as base64 image.
    """
    import base64
    
    model_name = model.split("ollama/", 1)[1] if "ollama/" in model else model
    
    # Try to extract screenshot path for vision models
    screenshot_path = None
    try:
        prompt_json = json.loads(prompt)
        screenshot_path = prompt_json.get("observation", {}).get("screenshot_path")
    except Exception:  # noqa: BLE001
        pass
    
    # Build request payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json"  # Force JSON output format
    }
    
    # If vision model and we have a screenshot, add it as base64 image
    if screenshot_path and Path(screenshot_path).exists():
        try:
            # Resize image to reduce processing time (vision models can be slow on large images)
            from PIL import Image
            import io
            
            img = Image.open(screenshot_path)
            
            # Convert RGBA to RGB (moondream needs RGB for JPEG)
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to fit within 640x640 box while maintaining aspect ratio
            max_dimension = 640
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to base64 (JPEG works, PNG causes empty responses)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Ollama vision API format
            payload["images"] = [img_b64]
            print(f"_call_ollama: sending image from {screenshot_path} (resized to {img.width}x{img.height}, JPEG)")
        except Exception as exc:  # noqa: BLE001
            print(f"_call_ollama: failed to load image {screenshot_path}: {exc}")
    
    try:
        import time as time_module
        start_time = time_module.time()
        print(f"_call_ollama: sending to Ollama (model={model_name}, timeout=60s)...")
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60,  # moondream should complete in <10s
        )
        elapsed = time_module.time() - start_time
        print(f"_call_ollama: received response in {elapsed:.1f}s (status={resp.status_code})")
    except Exception as exc:  # noqa: BLE001
        print(f"_call_ollama error: {exc}")
        return None

    if resp.status_code != 200:
        print(f"_call_ollama http {resp.status_code}: {resp.text[:200]}")
        return None
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"_call_ollama parse error: {exc}")
        return None
    
    response_text = data.get("response", "")
    print(f"_call_ollama: Ollama returned {len(response_text)} characters")
    return response_text


def make_adb_tools(
    device_id: Optional[str] = None,
    dry_run: bool = False,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Factory for ADB-backed tool callables that can be plugged into ADK LlmAgents.
    Each callable returns a dict that includes status and relevant metadata.
    """
    client = AdbClient(device_id=device_id, dry_run=dry_run)
    base_dir = artifacts_dir or Path("artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)

    def adb_screencap() -> Dict[str, Any]:
        """Take emulator screenshot; return path."""
        ts = int(time.time() * 1000)
        path = base_dir / f"obs_{ts}.png"
        res = client.screenshot(path)
        res["path"] = str(path)
        return res

    def adb_uia_dump() -> Dict[str, Any]:
        """Dump UI XML and return path."""
        path = base_dir / "runtime_uidump.xml"
        res = client.dump_ui(path)
        res["path"] = str(path)
        return res

    def adb_tap(x: int, y: int) -> Dict[str, Any]:
        """Tap absolute screen coordinates."""
        return client.tap(int(x), int(y))

    def adb_type(text: str) -> Dict[str, Any]:
        """Type text into the focused field."""
        return client.input_text(str(text))

    def adb_keyevent(code: int) -> Dict[str, Any]:
        """Send a keyevent code (e.g., 67 for backspace)."""
        return client.keyevent(int(code))

    return {
        "adb_screencap": adb_screencap,
        "adb_uia_dump": adb_uia_dump,
        "adb_tap": adb_tap,
        "adb_type": adb_type,
        "adb_keyevent": adb_keyevent,
    }


def build_llm_agents_with_tools(
    model: str = "gemini-2.5-flash",
    device_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, LlmAgent]:
    """
    Construct planner/executor/supervisor ADK agents wired with ADB tool wrappers
    and return them along with a root coordinator agent.
    """
    adb_tools = make_adb_tools(device_id=device_id, dry_run=dry_run)

    planner = LlmAgent(
        name="planner",
        model=model,
        description="Plans the next UI action for the current QA test step.",
        instruction=(
            "Given the test case and current UI state, output ONE next step as JSON:\n"
            "{action: ..., target: ..., args: ..., expected: ..., stop: bool}\n"
            "Prefer selecting elements from the UI tree. If an assertion is impossible, mark it as ASSERTION_FAILED."
        ),
        tools=[adb_tools["adb_screencap"], adb_tools["adb_uia_dump"]],
    )

    executor = LlmAgent(
        name="executor",
        model=model,
        description="Executes a planned step by calling ADB tools ONLY.",
        instruction=(
            "You receive a JSON plan. Execute using tools. "
            "If an element is missing, return ACTION_FAILED. "
            "Do NOT invent taps without coordinates or node bounds."
        ),
        tools=[
            adb_tools["adb_tap"],
            adb_tools["adb_type"],
            adb_tools["adb_keyevent"],
            adb_tools["adb_screencap"],
            adb_tools["adb_uia_dump"],
        ],
    )

    supervisor = LlmAgent(
        name="supervisor",
        model=model,
        description="Verifies state transitions and final pass/fail.",
        instruction=(
            "Decide PASS/FAIL for each step. "
            "Differentiate ACTION_FAILED vs ASSERTION_FAILED. "
            "For FAIL: produce a short bug-style report with evidence."
        ),
        tools=[adb_tools["adb_screencap"], adb_tools["adb_uia_dump"]],
    )

    root_agent = LlmAgent(
        name="root",
        model=model,
        description="Coordinates planner/executor/supervisor to run QA tests.",
        instruction="Run tests by delegating to sub-agents.",
        sub_agents=[planner, executor, supervisor],
    )

    return {
        "planner": planner,
        "executor": executor,
        "supervisor": supervisor,
        "root": root_agent,
    }


def _call_gemini(model: str, prompt: str) -> Optional[str]:
    """Lightweight REST call to Gemini; avoids needing client packages."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY_1")
    if not api_key:
        return None
    # Prefer provided model; fall back to a known public one if needed.
    candidate_models = [
        model,
        "gemini-2.5-flash",
    ]
    last_err: Optional[str] = None
    for m in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=20)
            if resp.status_code != 200:
                last_err = f"{m} -> {resp.status_code}: {resp.text[:200]}"
                continue
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                last_err = f"{m} -> no candidates"
                continue
            parts = candidates[0].get("content", {}).get("parts") or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            joined = "\n".join(t for t in texts if t)
            if joined:
                return joined
        except Exception as exc:  # noqa: BLE001
            last_err = f"{m} -> {exc}"
            continue
    if last_err:
        raise RuntimeError(f"Gemini call failed: {last_err}")
    return None


def plan_with_adk(planner: LlmAgent, test_case: Dict[str, Any], observation: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hybrid planning:
    1. Vision model (moondream) detects screen state via classify_screen
    2. Heuristic rules map state → actions (fast, reliable)
    3. Falls back to LLM for unknown states
    """
    # Use heuristic planning based on detected state (moondream classifies, heuristics plan)
    state = observation.get("state", "")
    visual_state = observation.get("visual_state", "")
    current_state = visual_state or state
    
    print(f"plan_with_adk: HYBRID MODE - state={current_state}")
    
    # Heuristic planning for known states
    if current_state == "create_vault_splash":
        print("plan_with_adk: HEURISTIC → tap 'Create a vault'")
        return [{"action": "tap_text", "detail": {"text": "Create a vault"}, "comment": "Initiate vault creation"}]
    
    elif current_state == "sync_dialog":
        print("plan_with_adk: HEURISTIC → tap 'Continue without sync'")
        return [{"action": "tap_text", "detail": {"text": "Continue without sync"}, "comment": "Skip sync setup"}]
    
    elif current_state == "storage_config":
        print("plan_with_adk: HEURISTIC → configure vault")
        goal = test_case.get("description", "")
        vault_name = "InternVault"  # Default name
        
        # Extract vault name from goal if present
        if "vault named" in goal.lower():
            import re
            match = re.search(r"named\s+['\"]([^'\"]+)['\"]", goal, re.IGNORECASE)
            if match:
                vault_name = match.group(1)
        
        return [
            {"action": "tap", "detail": {"x_norm": 0.5, "y_norm": 0.25}, "comment": "Focus vault name field"},
            {"action": "input_text", "detail": {"text": vault_name}, "comment": f"Enter vault name '{vault_name}'"},
            {"action": "tap_text", "detail": {"text": "Device storage"}, "comment": "Select device storage"},
            {"action": "wait", "detail": {"seconds": 0.5}, "comment": "Wait after selection"},
            {"action": "tap_text", "detail": {"text": "Create"}, "comment": "Finalize vault creation"},
        ]
    
    elif current_state == "editor":
        print("plan_with_adk: HEURISTIC → task complete (reached editor)")
        return [{"action": "wait", "detail": {"seconds": 1}, "comment": "Task complete - in editor"}]
    
    # Unknown state - fall back to LLM planning
    print(f"plan_with_adk: Unknown state '{current_state}', falling back to LLM...")
    
    # Full prompt for LLM fallback
    prompt_payload: Dict[str, Any] = {
        "test_case": test_case,
        "observation": observation,
        "history": history[-5:],  # keep prompt small
        "instructions": (
            "Return ONLY a JSON array of steps: [{\"action\": str, \"detail\": object, \"comment\": str}]. "
            "Allowed actions: start_app, stop_app, clear_app, tap, tap_text, swipe, input_text, keyevent, "
            "keycombination, wait, screenshot, dump_ui. "
            "For tapping buttons/text: use 'tap_text' with {\"text\": \"exact button text\"}. "
            "For typing text: use 'input_text' with {\"text\": \"text to type\"}. "
            "For precise coordinates: use 'tap' with {\"x_norm\": float, \"y_norm\": float} (0-1 normalized). "
            "Do NOT use 'click', 'type', or 'resource_id'. Keep 1-3 concise steps. No prose."
        ),
    }
    if observation.get("screenshot_path"):
        prompt_payload["visual_hint"] = (
            "A screenshot is available at observation.screenshot_path; reason over it to choose taps."
        )
    prompt = json.dumps(prompt_payload, ensure_ascii=False)
    
    print(f"plan_with_adk: prompt={prompt[:200]}...")
    try:
        text: Optional[str] = None

        # Prefer local Ollama when requested (e.g., "ollama/qwen2.5vl:7b")
        if isinstance(model_name, str) and model_name.startswith("ollama/"):
            print(f"plan_with_adk: calling Ollama model {model_name}...")
            text = _call_ollama(model_name, prompt)
            print(f"plan_with_adk: Ollama call completed, returned text: {text is not None}")
        else:
            text = _call_gemini(str(model_name), prompt)

        print(f"\n\nplan_with_adk response text: {text}")
        if text:
            print(f"plan_with_adk: parsing response...")
            # Strip markdown code fences if present (Ollama often wraps JSON)
            text_clean = text.strip()
            if text_clean.startswith("```"):
                # Remove ```json or ``` at start and ``` at end
                lines = text_clean.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text_clean = "\n".join(lines)
            
            steps = json.loads(text_clean)
            
            # Moondream returns single object, wrap in array
            if isinstance(steps, dict):
                steps = [steps]
            
            if isinstance(steps, list):
                # Normalize actions and details from the planner.
                for step in steps:
                    # Normalize action names
                    action = step.get("action", "")
                    if action == "click":
                        step["action"] = "tap"
                    elif action in ("type", "enter_text", "input"):
                        step["action"] = "input_text"
                    
                    detail = step.get("detail") or {}
                    print(f"\n\ndetail: {detail}")
                    if isinstance(detail, dict):
                        # If tap action has "text" field, convert to tap_text
                        if step.get("action") == "tap" and "text" in detail and "x" not in detail and "x_norm" not in detail:
                            step["action"] = "tap_text"
                        
                        # If detail has resource_id but no coordinates, try to use tap_text
                        # (executor doesn't support resource_id directly)
                        if "resource_id" in detail and "x" not in detail and "x_norm" not in detail:
                            # For create_vault_button, use known button text
                            if "create_vault" in detail.get("resource_id", "").lower():
                                step["action"] = "tap_text"
                                step["detail"] = {"text": "Create a vault"}
                            else:
                                # Generic fallback: remove resource_id and use center tap
                                detail.pop("resource_id", None)
                                if not detail:  # If detail is now empty, use center screen
                                    step["detail"] = {"x_norm": 0.5, "y_norm": 0.5}
                        
                        # Normalize button text and input values
                        raw_text = detail.get("text")
                        if isinstance(raw_text, str):
                            text_lower = raw_text.strip().lower()
                            # Normalize vault creation button
                            if text_lower == "create new vault":
                                detail["text"] = "Create a vault"
                                step["detail"] = detail
                            # Normalize sync dialog button (handle variations)
                            elif "continue without sync" in text_lower or "skip sync" in text_lower or "no sync" in text_lower:
                                detail["text"] = "Continue without sync"
                                step["detail"] = detail
                            # Normalize storage options
                            elif "device storage" in text_lower or "local storage" in text_lower:
                                detail["text"] = "Device storage"
                                step["detail"] = detail
                            # Keep vault name exactly as specified in test case
                            elif step.get("action") == "input_text" and "internvault" in text_lower:
                                detail["text"] = "InternVault"
                                step["detail"] = detail
                print(f"\n\nplan_with_adk final plan: {steps}")
                print(f"plan_with_adk: returning {len(steps)} steps")
                return steps
    except Exception as e:
        # Ignore and fall through to fallback.
        print(f"plan_with_adk: exception during parsing: {e}")
        pass
    print("plan_with_adk: using fallback plan (LLM planning failed)")
    return [
        {
            "action": "interpret_goal",
            "detail": test_case.get("description", ""),
            "comment": "Fallback plan from Gemini planning failure.",
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

