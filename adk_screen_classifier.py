from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    from google.adk.agents import LlmAgent
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "google-adk is required. Install with `pip install google-adk`."
    ) from exc


def build_screen_classifier(model: str = "gemini-2.5-flash") -> LlmAgent:
    """
    Lightweight classifier: given a screenshot path, return a label from:
    create_vault_splash | sync_dialog | storage_config | editor | unknown
    """
    return LlmAgent(
        name="screen_classifier",
        model=model,
        instruction=(
            "You classify mobile screenshots. Return ONLY one of:\n"
            "create_vault_splash | sync_dialog | storage_config | editor | unknown\n"
            "Use: splash with purple 'Create a vault'; sync dialog with CTA 'Continue without sync';\n"
            "storage_config shows 'Vault name' and storage options; editor shows note UI.\n"
            "Input is a JSON with screenshot_path."
        ),
    )


def classify_screen(classifier: LlmAgent, screenshot_path: Path) -> str:
    if not screenshot_path or not screenshot_path.exists():
        return "unknown"
    prompt = {"screenshot_path": str(screenshot_path)}
    try:
        raw = classifier(prompt)
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict) and "output" in raw:
            return str(raw["output"]).strip()
        return str(raw).strip()
    except Exception:
        return "unknown"

