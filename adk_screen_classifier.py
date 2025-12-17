from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests

try:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "google-adk is required. Install with `pip install google-adk`."
    ) from exc


def build_screen_classifier(model: str = "ollama/moondream") -> LlmAgent:
    """
    Lightweight classifier: given a screenshot path, return a label from:
    create_vault_splash | sync_dialog | storage_config | editor | unknown
    Uses local Moondream model via Ollama by default (fast, lightweight vision model).
    """
    # Use LiteLlm wrapper for Ollama models
    if isinstance(model, str) and model.startswith("ollama/"):
        model_obj = LiteLlm(model=model)
    else:
        model_obj = model
    
    return LlmAgent(
        name="screen_classifier",
        model=model_obj,
        instruction=(
            "Describe what you see on this mobile app screen. "
            "What buttons, text, or UI elements are visible?"
        ),
    )


def _call_model_with_instruction(model: str, instruction: str, payload: Dict[str, str]) -> Optional[str]:
    """
    Minimal REST helper to call either Ollama (for ollama/... models) or Gemini.
    Includes the instruction and the JSON payload (e.g., screenshot_path).
    """
    # Check if using Ollama
    if model.startswith("ollama/"):
        import base64
        from pathlib import Path as PathLib
        
        model_name = model.replace("ollama/", "")
        
        # For vision models with images, use ONLY the instruction (no JSON clutter)
        screenshot_path = payload.get("screenshot_path")
        if screenshot_path and PathLib(screenshot_path).exists():
            prompt_text = instruction  # Simple prompt only
        else:
            prompt_text = f"{instruction}\nInput:\n{json.dumps(payload, ensure_ascii=False)}"
        
        body = {"model": model_name, "prompt": prompt_text, "stream": False}
        
        # Extract screenshot path and add image for vision models
        screenshot_path = payload.get("screenshot_path")
        if screenshot_path and PathLib(screenshot_path).exists():
            try:
                # Resize image to reduce processing time
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
                
                body["images"] = [img_b64]
                print(f"classify_screen: sending image from {screenshot_path} (resized to {img.width}x{img.height}, JPEG)")
            except Exception as exc:  # noqa: BLE001
                print(f"classify_screen: failed to load image {screenshot_path}: {exc}")
        
        try:
            url = "http://localhost:11434/api/generate"
            print(f"_call_model_with_instruction: sending request to Ollama (model={model_name}, timeout=60s)...")
            resp = requests.post(url, json=body, timeout=60)  # Moondream is fast
            print(f"_call_model_with_instruction: received response from Ollama (status={resp.status_code})")
            if resp.status_code != 200:
                print(f"classify_screen: Ollama call failed: {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()
            response = data.get("response", "").strip()
            print(f"_call_model_with_instruction: Ollama returned {len(response)} characters")
            # Strip markdown code fences if present
            if response.startswith("```"):
                lines = response.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response = "\n".join(lines).strip()
            return response
        except Exception as exc:  # noqa: BLE001
            print(f"_call_model_with_instruction: Ollama error: {exc}")
            return None
    


def classify_screen(classifier: LlmAgent, screenshot_path: Path) -> str:
    print(f"classify_screen: START - processing {screenshot_path}")
    if not screenshot_path or not screenshot_path.exists():
        print(f"classify_screen: screenshot not found, returning unknown")
        return "unknown"
    
    prompt = {"screenshot_path": str(screenshot_path)}
    print(f"classify_screen: trying to call LlmAgent directly...")
    try:
        # Try to call LlmAgent directly (some ADK versions support this)
        raw = classifier(prompt)
        print(f"classify_screen raw response (callable): {raw}")
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict) and "output" in raw:
            return str(raw["output"]).strip()
        return str(raw).strip()
    except (TypeError, AttributeError) as e:  # noqa: BLE001
        # LlmAgent not callable in this ADK version, fall back to REST
        print(f"classify_screen: LlmAgent not callable ({e}), falling back to REST")
        pass  # Silently fall back to REST (expected behavior)

    # Fallback: direct REST call (Ollama or Gemini) using the agent's instruction and model.
    print(f"classify_screen: extracting model info from classifier...")
    instruction = getattr(classifier, "instruction", "")
    model = getattr(classifier, "model", "ollama/moondream")
    # Extract model string if it's a LiteLlm object
    if hasattr(model, "model"):
        model = model.model
    print(f"classify_screen: calling _call_model_with_instruction with model={model}")
    text = _call_model_with_instruction(str(model), str(instruction), prompt)
    if text:
        print(f"classify_screen raw response (REST): {text[:200]}")
        # Map description keywords to states
        text_lower = text.lower()
        # Vault splash: mentions vault + button/purple (initial screen)
        if ("vault" in text_lower and ("button" in text_lower or "purple" in text_lower)) or \
           ("create" in text_lower and "vault" in text_lower):
            result = "create_vault_splash"
        # Sync dialog: mentions sync or continue
        elif "sync" in text_lower or "continue without" in text_lower or "continue" in text_lower:
            result = "sync_dialog"
        # Storage config: mentions storage or configuration fields
        elif "storage" in text_lower or "vault name" in text_lower or "device storage" in text_lower or \
             ("name" in text_lower and "field" in text_lower):
            result = "storage_config"
        # Editor: mentions editing, notes, markdown
        elif "editor" in text_lower or "note" in text_lower or "markdown" in text_lower or "typing" in text_lower:
            result = "editor"
        else:
            result = "unknown"
        print(f"classify_screen: DONE - mapped to '{result}'")
        return result
    print(f"classify_screen: DONE - no response, returning unknown")
    return "unknown"

