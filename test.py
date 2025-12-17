from agents import Executor

# Use adb for real device/emulator; switch to "dry" to simulate.
executor = Executor(mode="adb", device_id=None)

observation = {
    "state": "vault_splash",
    "screenshot_path": "artifacts/obs_0.png",
    "dump_path": None,
}

test_case = {
    "id": "T1",
    "description": "Tap the Create a vault button on splash.",
    "expected_outcome": "pass",
}

# Deterministic plan: tap the button label directly.
plan = [
    {
        "action": "tap_text",
        "detail": {"text": "Create a vault"},
        "comment": "Tap the Create a vault button on the splash screen.",
    }
]

result = executor.execute(plan, test_case, observation)
print(result)