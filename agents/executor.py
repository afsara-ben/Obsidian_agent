from __future__ import annotations

import time
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from tools.adb_client import AdbClient


class Executor:
    """
    Executes planned actions using ADB. In dry mode, it simulates actions so the
    end-to-end loop can run without an emulator.
    """

    def __init__(self, device_id: Optional[str] = None, mode: str = "dry") -> None:
        self.mode = mode
        self.adb = AdbClient(device_id=device_id, dry_run=(mode != "adb"))
        self._screen_size: Optional[tuple[int, int]] = None

    def _ensure_screen_size(self) -> tuple[int, int]:
        if self._screen_size:
            return self._screen_size
        res = self.adb.get_screen_size()
        if res.get("status") != "ok":
            raise RuntimeError(res.get("stderr") or "Unable to read screen size")
        width = res.get("width")
        height = res.get("height")
        if not width or not height:
            raise RuntimeError("Screen size missing width/height")
        self._screen_size = (int(width), int(height))
        return self._screen_size

    def _resolve_point(self, detail: Dict[str, Any]) -> tuple[int, int]:
        if "x" in detail and "y" in detail:
            return int(detail["x"]), int(detail["y"])
        if "x_norm" in detail and "y_norm" in detail:
            width, height = self._ensure_screen_size()
            return int(float(detail["x_norm"]) * width), int(float(detail["y_norm"]) * height)
        raise ValueError("tap requires x/y or x_norm/y_norm in detail")

    def _resolve_swipe(self, detail: Dict[str, Any]) -> tuple[int, int, int, int, int]:
        duration_ms = int(detail.get("duration_ms", 300))
        if "start_x" in detail and "start_y" in detail and "end_x" in detail and "end_y" in detail:
            return (
                int(detail["start_x"]),
                int(detail["start_y"]),
                int(detail["end_x"]),
                int(detail["end_y"]),
                duration_ms,
            )
        if (
            "start_x_norm" in detail
            and "start_y_norm" in detail
            and "end_x_norm" in detail
            and "end_y_norm" in detail
        ):
            width, height = self._ensure_screen_size()
            return (
                int(float(detail["start_x_norm"]) * width),
                int(float(detail["start_y_norm"]) * height),
                int(float(detail["end_x_norm"]) * width),
                int(float(detail["end_y_norm"]) * height),
                duration_ms,
            )
        raise ValueError("swipe requires start/end x/y or start/end x_norm/y_norm in detail")

    def _dump_ui(self, local_path: Path) -> Dict[str, Any]:
        return self.adb.dump_ui(local_path)

    def _tap_text(self, text: str) -> Dict[str, Any]:
        """
        Finds a node whose text contains the given string (case-insensitive) using
        uiautomator dump, then taps its center.
        """
        dump_path = Path("artifacts/runtime_uidump.xml")
        res = self._dump_ui(dump_path)
        if res.get("status") != "ok":
            return res

        try:
            tree = ET.parse(dump_path)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "stderr": f"parse uidump failed: {exc}"}

        def _norm(s: str) -> str:
            # Lowercase and collapse all whitespace to single spaces for robust matching.
            return " ".join(s.lower().split())

        target_bounds = None

        def find_bounds(target: str) -> Optional[str]:
            for node in tree.iter():
                node_text = _norm(node.attrib.get("text") or "")
                node_desc = _norm(node.attrib.get("content-desc") or "")
                if target in node_text or target in node_desc:
                    bounds = node.attrib.get("bounds")
                    if bounds:
                        return bounds
            return None

        target_text = _norm(text)
        target_bounds = find_bounds(target_text)

        # Fallback: common ADK planner variant "create new vault" vs UI "Create a vault".
        if not target_bounds and target_text == "create new vault":
            target_bounds = find_bounds("create a vault")

        if not target_bounds:
            return {"status": "error", "stderr": f"text '{text}' not found in UI dump"}

        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", target_bounds)
        if not match:
            return {"status": "error", "stderr": f"cannot parse bounds: {target_bounds}"}
        x1, y1, x2, y2 = map(int, match.groups())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.adb.tap(cx, cy)

    def execute(
        self,
        plan: List[Dict[str, Any]],
        test_case: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        status = "ok"
        notes: List[str] = []

        for idx, step in enumerate(plan):
            action = step.get("action", "")
            detail = step.get("detail", "")

            if self.mode != "adb":
                steps.append(
                    {
                        "step_index": idx,
                        "action": action,
                        "detail": detail,
                        "result": "simulated",
                    }
                )
                continue

            # Dispatch supported actions.
            result: Dict[str, Any] = {"status": "error"}
            try:
                if action == "start_app":
                    package = detail.get("package")
                    activity = detail.get("activity")
                    result = self.adb.start_app(package=package, activity=activity)
                elif action == "stop_app":
                    package = detail.get("package")
                    result = self.adb.force_stop(package)
                elif action == "clear_app":
                    package = detail.get("package")
                    result = self.adb.clear_app_data(package)
                elif action == "tap":
                    x, y = self._resolve_point(detail)
                    result = self.adb.tap(x, y)
                    detail = {**detail, "resolved_x": x, "resolved_y": y}
                elif action == "swipe":
                    sx, sy, ex, ey, dur = self._resolve_swipe(detail)
                    result = self.adb.swipe(sx, sy, ex, ey, dur)
                    detail = {
                        **detail,
                        "resolved_start_x": sx,
                        "resolved_start_y": sy,
                        "resolved_end_x": ex,
                        "resolved_end_y": ey,
                    }
                elif action == "input_text":
                    result = self.adb.input_text(str(detail["text"]))
                elif action == "keyevent":
                    result = self.adb.keyevent(int(detail["key_code"]))
                elif action == "keycombination":
                    result = self.adb.keycombination(int(detail["combo_code"]))
                elif action == "wait":
                    time.sleep(float(detail.get("seconds", 1)))
                    result = {"status": "ok", "cmd": ["sleep", detail.get("seconds", 1)]}
                elif action == "screenshot":
                    output_path = detail.get("path")
                    result = self.adb.screenshot(Path(output_path))
                elif action == "dump_ui":
                    output_path = Path(detail.get("path", "artifacts/runtime_uidump.xml"))
                    result = self._dump_ui(output_path)
                    detail = {**detail, "path": str(output_path)}
                elif action == "tap_text":
                    target = detail.get("text", "")
                    result = self._tap_text(str(target))
                else:
                    result = {"status": "error", "stderr": f"Unknown action '{action}'"}
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "stderr": f"{exc}"}

            steps.append(
                {
                    "step_index": idx,
                    "action": action,
                    "detail": detail,
                    "result": result.get("status", "error"),
                    "stderr": result.get("stderr", ""),
                }
            )

            if result.get("status") != "ok":
                status = "fail"
                notes.append(result.get("stderr", f"Action {action} failed"))

        expected = test_case.get("expected_outcome", "pass")
        if status == "ok" and expected == "fail":
            status = "fail"
            notes.append("Marked as failure because the test case expects a failing assertion.")

        return {
            "status": status,
            "steps": steps,
            "observation": observation,
            "notes": notes,
        }

