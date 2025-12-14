from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class AdbClient:
    """
    Thin wrapper around adb to keep command construction consistent.
    Supports a dry-run mode for environments without an attached emulator.
    """

    def __init__(self, device_id: Optional[str] = None, dry_run: bool = False) -> None:
        self.device_id = device_id
        self.dry_run = dry_run

    def _cmd(self, args: List[str]) -> List[str]:
        base = ["adb"]
        if self.device_id:
            base.extend(["-s", self.device_id])
        base.extend(args)
        return base

    def _run(
        self,
        args: List[str],
        timeout: int = 10,
        capture_output: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        cmd = self._cmd(args)
        if self.dry_run:
            return {"cmd": cmd, "status": "simulated"}

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=capture_output,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"cmd": cmd, "status": "timeout"}

        if output_path and result.stdout is not None:
            output_path.write_bytes(result.stdout)

        status = "ok" if result.returncode == 0 else "error"
        return {
            "cmd": cmd,
            "status": status,
            "stdout": result.stdout.decode("utf-8", errors="ignore") if result.stdout else "",
            "stderr": result.stderr.decode("utf-8", errors="ignore") if result.stderr else "",
            "returncode": result.returncode,
        }

    def tap(self, x: int, y: int) -> Dict[str, Any]:
        return self._run(["shell", "input", "tap", str(x), str(y)])

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300) -> Dict[str, Any]:
        return self._run(
            ["shell", "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), str(duration_ms)]
        )

    def keyevent(self, key_code: int) -> Dict[str, Any]:
        return self._run(["shell", "input", "keyevent", str(key_code)])

    def keycombination(self, combo_code: int) -> Dict[str, Any]:
        """
        Android 13+ supports keycombination for select-all (e.g., 11329 for Ctrl+A).
        """
        return self._run(["shell", "input", "keycombination", str(combo_code)])

    def input_text(self, text: str) -> Dict[str, Any]:
        safe_text = text.replace(" ", "%s")
        return self._run(["shell", "input", "text", safe_text])

    def screenshot(self, output_path: Path) -> Dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._run(["exec-out", "screencap", "-p"], capture_output=True, output_path=output_path)

    def dump_ui(self, local_path: Path) -> Dict[str, Any]:
        """
        Dumps the current UI hierarchy to /sdcard/uidump.xml and pulls it to local_path.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        res = self._run(["shell", "uiautomator", "dump", "/sdcard/uidump.xml"])
        if res.get("status") != "ok":
            return res
        pull = self._run(["pull", "/sdcard/uidump.xml", str(local_path)])
        return pull

    def start_app(self, package: str, activity: Optional[str] = None) -> Dict[str, Any]:
        if activity:
            # Accept either a full component ("pkg/.Activity") or a short ".Activity".
            if "/" in activity:
                component = activity
            else:
                component = f"{package}/{activity}"
            return self._run(["shell", "am", "start", "-n", component])
        return self._run(["shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"])

    def force_stop(self, package: str) -> Dict[str, Any]:
        return self._run(["shell", "am", "force-stop", package])

    def clear_app_data(self, package: str) -> Dict[str, Any]:
        return self._run(["shell", "pm", "clear", package])

    def get_screen_size(self) -> Dict[str, Any]:
        """
        Returns {"status": "ok", "width": int, "height": int, ...} or an error status.
        """
        res = self._run(["shell", "wm", "size"])
        if res.get("status") != "ok":
            return res
        out = res.get("stdout", "")
        match = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
        if not match:
            res["status"] = "error"
            res["stderr"] = f"Unable to parse screen size from: {out}"
            return res
        res["width"] = int(match.group(1))
        res["height"] = int(match.group(2))
        return res

