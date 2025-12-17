# Obsidian Agent - Intelligent Android UI Automation

An intelligent multi-agent system for automating Android UI interactions using vision-based screen classification and rule-based planning. Built with local Ollama vision models and custom ADB integration.

## 🎯 Overview

This project demonstrates a **hybrid AI approach** to Android UI automation:

- **Vision AI** (Moondream via Ollama) for screen state detection
- **Heuristic Planning** for reliable action generation
- **ADB Integration** for device control
- **Multi-Agent Architecture** for modular task execution

**Example Use Case:** Automates creating a vault in the Obsidian Android app - detecting screens, filling forms, and navigating through multiple UI states.

https://github.com/user-attachments/assets/55b3bf7f-b972-4c0c-8169-67529ce429ec
## 🔑 Core Technologies

- **Ollama** - Local LLM server hosting Moondream vision model
- **Moondream** - 280M parameter vision-language model for screen classification
- **ADB (Android Debug Bridge)** - Direct device control via shell commands
- **Python 3.12+** - Modern async/await patterns and type hints

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVATION CAPTURE                      │
│  • Screenshot (ADB)                                         │
│  • UI Dump XML (uiautomator)                                │
│  • Clickable Elements Extraction                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               VISION CLASSIFICATION (~2s)                   │
│  • Moondream LLM (local via Ollama)                         │
│  • Screen → "vault_splash", "sync_dialog", etc.             │
│  • Fallback: Heuristic XML parsing                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              HEURISTIC PLANNING (instant)                   │
│  • State-based rules                                        │
│  • Deterministic action generation                          │
│  • Multi-step plans for complex UIs                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  EXECUTION (ADB)                            │
│  • tap / tap_text                                           │
│  • input_text                                               │
│  • swipe / keyevent                                         │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 1. **Hybrid Vision + Heuristic Approach**
- **Vision Model**: Moondream (280M params) for semantic screen understanding
- **Heuristic Fallback**: Fast XML parsing for known UI patterns
- **State Priority**: Prefers deterministic heuristics when confident

### 2. **Multi-Agent System**
- **Planner**: Generates action sequences based on screen state
- **Executor**: Executes ADB commands with error handling
- **Supervisor**: Monitors execution and validates outcomes
- **Screen Classifier**: Identifies screen types from screenshots

### 3. **Robust Action Execution**
- **tap_text**: Finds UI elements by text, calculates center, taps
- **Smart Coordinates**: Extracts bounds from XML, handles normalized coords
- **Error Recovery**: Fallback plans, retry logic, detailed logging

### 4. **State Detection**

| State | Detection Method | Trigger Keywords |
|-------|------------------|------------------|
| `vault_splash` | XML parsing | "create a vault" + "existing vault" |
| `sync_dialog` | XML parsing | "continue without sync" |
| `vault_config` | XML parsing | "configure your new vault", "vault name" |
| `editor` | Vision/Heuristic | "untitled" note screen |

## 🚀 Setup

### Prerequisites

```bash
# 1. Python 3.12+
python --version

# 2. Android SDK Platform Tools (ADB)
adb version

# 3. Ollama (for local vision model)
ollama --version

# 4. Connected Android device/emulator
adb devices
```

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd Obsidian_agent

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull Ollama vision model
ollama pull moondream

# 4. Optional: Set up Gemini API (alternative to Ollama)
export GOOGLE_API_KEY="your-api-key"
```

### Dependencies

```txt
requests>=2.31.0   # HTTP client for Ollama API
Pillow>=10.0.0     # Image processing (resize, JPEG conversion)
```

No heavy ML frameworks needed - everything runs via Ollama's REST API!

## 📱 Usage

### Basic Usage

```bash
# Run vault creation automation
python run_adk.py \
  --mode adb \
  --device-id emulator-5556 \
  --tests tests/test_cases.json
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tests` | Path to test cases JSON | `tests/test_cases.json` |
| `--device-id` | ADB device identifier | Auto-detect |
| `--mode` | Execution mode: `adb` or `dry` | `adb` |
| `--model` | Vision model | `ollama/moondream` |
| `--max-steps` | Max steps per test | `20` |

### Test Case Format

```json
[
  {
    "id": "T1",
    "description": "Create a new Vault named 'InternVault'",
    "expected_outcome": "pass",
    "script": [
      {
        "action": "start_app",
        "detail": {
          "package": "md.obsidian",
          "activity": ".MainActivity"
        }
      }
    ]
  }
]
```

## 🧠 How It Works

### 1. Screen Classification

```python
# Vision-based (Moondream via Ollama)
visual_state = classify_screen(classifier, screenshot_path)
# → "create_vault_splash"

# Heuristic-based (XML parsing)
state = detect_state(adb_client, ui_dump_path)
# → "vault_config"

# Priority: Heuristic > Vision (when heuristic is confident)
```

### 2. Planning (State → Actions)

```python
# State: vault_splash
plan = [
    {"action": "tap_text", "detail": {"text": "Create a vault"}}
]

# State: vault_config
plan = [
    {"action": "tap", "detail": {"x_norm": 0.5, "y_norm": 0.283}},
    {"action": "input_text", "detail": {"text": "InternVault"}},
    {"action": "tap_text", "detail": {"text": "Device storage"}},
    {"action": "tap_text", "detail": {"text": "Create a vault"}}
]
```

### 3. Execution

```python
# tap_text: Finds element in UI dump, calculates center, taps
executor.execute_step({
    "action": "tap_text",
    "detail": {"text": "Create a vault"}
})
# → Searches XML for text="Create a vault"
# → Extracts bounds="[84,1055][997,1165]"
# → Calculates center (540, 1110)
# → Executes: adb shell input tap 540 1110
```

## 📊 Supported Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `tap` | `x_norm`, `y_norm` OR `x`, `y` | Tap at coordinates |
| `tap_text` | `text` | Find element by text and tap |
| `input_text` | `text` | Type text into focused field |
| `swipe` | `start_x/y`, `end_x/y`, `duration_ms` | Swipe gesture |
| `keyevent` | `key_code` | Send Android key code |
| `wait` | `seconds` | Pause execution |
| `start_app` | `package`, `activity` | Launch app |
| `screenshot` | - | Capture screen |
| `dump_ui` | - | Get UI hierarchy XML |

## 🔧 Configuration

### Vision Model Settings

```python
# adk_screen_classifier.py
# Direct Ollama REST API call
def _call_model_with_instruction(model: str, instruction: str, payload: Dict):
    url = "http://localhost:11434/api/generate"
    body = {
        "model": "moondream",
        "prompt": instruction,
        "stream": False,
        "format": "json",
        "images": [base64_encoded_image]
    }
    response = requests.post(url, json=body, timeout=60)
    return response.json()["response"]
```

### Heuristic Planning Rules

```python
# adk_agents.py - Add custom state handlers
if current_state == "custom_screen":
    return [
        {"action": "tap_text", "detail": {"text": "Button Name"}},
        {"action": "wait", "detail": {"seconds": 1}}
    ]
```

## 🐛 Troubleshooting

### Issue: Moondream Returns Empty Responses

**Cause:** PNG format not supported well by moondream  
**Fix:** Images are automatically converted to JPEG (quality 85)

```python
# Image is resized to 640x640 and converted to JPEG
img.save(buffer, format="JPEG", quality=85)
```

### Issue: Text Field Not Written

**Cause:** Wrong tap coordinates  
**Fix:** Verify textfield bounds in UI dump

```bash
# Get UI dump
adb shell uiautomator dump /sdcard/ui.xml
adb pull /sdcard/ui.xml

# Find EditText element, check bounds attribute
grep -i "EditText" ui.xml
```

### Issue: ADB Device Not Found

```bash
# Check connected devices
adb devices

# Restart ADB server
adb kill-server
adb start-server

# Connect to emulator
adb connect localhost:5554
```

## 📁 Project Structure

```
Obsidian_agent/
├── adk_agents.py           # Multi-agent definitions & planning
├── adk_screen_classifier.py # Vision-based screen classification
├── run_adk.py              # Main entry point
├── agents/
│   ├── executor.py         # Action execution engine
│   └── supervisor.py       # Execution monitoring
├── tools/
│   ├── adb_client.py       # ADB wrapper
│   └── __init__.py
├── tests/
│   └── test_cases.json     # Test case definitions
├── artifacts/
│   ├── obs_*.png           # Screenshots
│   └── state_uidump.xml    # UI hierarchy dumps
└── requirements.txt
```

## 🎓 Key Learnings

### Why Hybrid (Vision + Heuristic)?

1. **Vision Models**: Great at semantic understanding, but:
   - Can misclassify similar screens
   - Struggle with JSON generation
   - Slower (~2s per screen)

2. **Heuristic Rules**: Fast and accurate, but:
   - Require manual rule creation
   - Break with UI changes

3. **Hybrid Approach**: Best of both worlds:
   - Use heuristics for known patterns (instant, 100% accurate)
   - Fall back to vision for unknown screens
   - Vision validates heuristic detection

### Why Moondream?

- **Lightweight**: 280M params vs 7B+ for alternatives
- **Fast**: ~2s inference on CPU
- **Good Enough**: Adequate for screen classification
- **Local**: No API costs, privacy preserved

