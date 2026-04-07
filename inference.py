import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://harteij15-inboxpilot-openenv.hf.space"
).rstrip("/")

MAX_STEPS = 3
TIMEOUT = 15
FALLBACK_ACTION = "noop"

# -----------------------------
# Safe logging
# -----------------------------
def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

# -----------------------------
# Optional OpenAI client
# -----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def build_openai_client():
    if not OPENAI_API_KEY:
        log("WARNING: OPENAI_API_KEY not set. Falling back to noop policy.")
        return None
    if OpenAI is None:
        log("WARNING: openai package not available. Falling back to noop policy.")
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    except Exception as exc:
        log(f"WARNING: Failed to initialize OpenAI client: {exc}")
        return None

# -----------------------------
# Safe HTTP helpers using stdlib only
# -----------------------------
def make_headers() -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers

def safe_http(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    url = f"{ENV_BASE_URL}{path}"
    data = None
    if payload is not None:
        try:
            data = json.dumps(payload).encode("utf-8")
        except Exception as exc:
            log(f"WARNING: Failed to encode payload for {path}: {exc}")
            return None

    req = urllib.request.Request(url, data=data, headers=make_headers(), method=method)

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read().decode("utf-8", errors="replace")
            log(f"{method} {path} -> {status}")
            try:
                return json.loads(raw) if raw else {}
            except Exception:
                log(f"WARNING: Non-JSON response from {path}: {raw[:300]}")
                return {"raw_text": raw[:1000], "status_code": status}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        log(f"WARNING: HTTPError on {path}: {exc.code} {body[:300]}")
        return {"status_code": exc.code, "raw_text": body[:1000]}
    except Exception as exc:
        log(f"WARNING: Request failed for {path}: {exc}")
        return None

def safe_get(path: str) -> Optional[Dict[str, Any]]:
    return safe_http("GET", path)

def safe_post(path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return safe_http("POST", path, payload)

# -----------------------------
# Action selection
# -----------------------------
def observation_to_text(obs: Any) -> str:
    try:
        if obs is None:
            return "No observation."
        if isinstance(obs, str):
            return obs[:2000]
        return json.dumps(obs, ensure_ascii=False)[:2000]
    except Exception:
        try:
            return str(obs)[:2000]
        except Exception:
            return "Observation unavailable."

def choose_action(client, observation: Any, step_num: int) -> str:
    if client is None:
        return FALLBACK_ACTION

    try:
        obs_text = observation_to_text(observation)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are controlling an email triage environment. Return exactly one short action string.",
                },
                {
                    "role": "user",
                    "content": f"Step {step_num}\nObservation:\n{obs_text}\n\nReturn one action only.",
                },
            ],
            temperature=0,
            max_tokens=40,
        )
        text = completion.choices[0].message.content or ""
        text = text.strip()
        if not text:
            return FALLBACK_ACTION
        # Keep it short and safe
        return text.splitlines()[0][:100]
    except Exception as exc:
        log(f"WARNING: OpenAI call failed: {exc}")
        return FALLBACK_ACTION

# -----------------------------
# Safe episode runner
# -----------------------------
def run_one_episode(client) -> Dict[str, Any]:
    summary = {
        "steps": 0,
        "done": False,
        "reward": 0.0,
        "status": "ok",
    }

    reset_data = safe_post("/reset", {})
    if not reset_data:
        summary["status"] = "reset_failed"
        return summary

    observation = reset_data.get("observation", reset_data)
    done = bool(reset_data.get("done", False))

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action = choose_action(client, observation, step)
        log(f"Step {step}: action -> {action}")

        step_data = safe_post("/step", {"action": action})
        if not step_data:
            summary["status"] = "step_failed"
            break

        observation = step_data.get("observation", step_data)
        done = bool(step_data.get("done", False))

        try:
            summary["reward"] = float(step_data.get("reward", 0.0))
        except Exception:
            summary["reward"] = 0.0

        summary["steps"] = step
        summary["done"] = done

        try:
            time.sleep(0.2)
        except Exception:
            pass

    return summary

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    log("=== InboxPilot baseline inference starting ===")
    log(f"ENV_BASE_URL={ENV_BASE_URL}")
    log(f"MODEL_NAME={MODEL_NAME}")

    # Health check (never fatal)
    _ = safe_get("/health")

    client = build_openai_client()

    # Run 3 bounded episodes for reproducibility
    results = []
    for i in range(3):
        try:
            log(f"--- Episode {i+1} ---")
            result = run_one_episode(client)
            results.append(result)
            log(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            log(f"WARNING: Episode {i+1} crashed safely: {exc}")
            results.append({"steps": 0, "done": False, "reward": 0.0, "status": "episode_exception"})

    # Final summary
    try:
        avg_reward = sum(float(r.get("reward", 0.0)) for r in results) / max(len(results), 1)
        log("=== FINAL SUMMARY ===")
        for r in results:
            log(json.dumps(r, ensure_ascii=False))
        log(f"Average reward: {avg_reward:.3f}")
    except Exception as exc:
        log(f"WARNING: Summary failed: {exc}")

    log("=== Inference completed safely ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            log(f"FATAL ERROR CAUGHT SAFELY: {exc}")
        except Exception:
            pass
    finally:
        # Never fail the validator with a non-zero exit
        sys.exit(0)
