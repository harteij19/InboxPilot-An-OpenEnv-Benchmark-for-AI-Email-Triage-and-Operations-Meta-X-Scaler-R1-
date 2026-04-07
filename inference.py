#!/usr/bin/env python3
"""
InboxPilot — Baseline inference script.

Evaluates all 3 tasks using an LLM via the OpenAI Python client.
Produces reproducible baseline scores with deterministic prompting (temperature=0).

IMPORTANT:
This script prints structured validator blocks:
- [START]
- [STEP]
- [END]
"""

from __future__ import annotations

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Add project root to path so we can import app.*
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL   = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
MAX_STEPS      = 60  # safety cap across all tasks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are InboxPilot, an AI email operations assistant. You must triage emails by taking structured actions.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation, no extra text.

Available actions:
- open_email
- classify_email
- set_priority
- draft_reply
- send_reply
- escalate
- mark_spam
- archive
- schedule_followup
- request_more_info
- finish

Response format (ONLY valid JSON):
{
  "action_type": "<action_name>",
  "email_id": "<email_id or null>",
  "payload": { ... }
}
"""


def build_user_prompt(observation: dict) -> str:
    parts = []
    parts.append(f"GOAL: {observation.get('goal', '')}")
    parts.append(f"INSTRUCTION: {observation.get('instruction', '')}")
    parts.append(f"Step: {observation.get('step_count', 0)}/{observation.get('max_steps', 50)}")
    parts.append(f"Pending emails: {observation.get('pending_items', [])}")

    inbox = observation.get("inbox_summary", [])
    if inbox:
        parts.append("\nINBOX:")
        for e in inbox:
            status = "open" if e.get("is_read") else "unread"
            cls  = e.get("classification", "unclassified")
            prio = e.get("priority", "unset")
            parts.append(
                f"  [{status}] [{e['email_id']}] From: {e['sender']} | "
                f"Subject: {e['subject']} | Class: {cls} | Priority: {prio}"
            )

    current = observation.get("current_email")
    if current:
        parts.append(f"\nCURRENT EMAIL [{current['email_id']}]:")
        parts.append(f"  From: {current['sender']}")
        parts.append(f"  Subject: {current['subject']}")
        parts.append(f"  Body: {current['body'][:500]}")

    history = observation.get("action_history", [])
    if history:
        parts.append("\nRECENT ACTIONS:")
        for h in history[-5:]:
            parts.append(
                f"  Step {h.get('step', '?')}: {h.get('action_type', '?')} "
                f"on {h.get('email_id', '-')}"
            )

    parts.append("\nRespond with ONLY a JSON object for your next action.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> dict:
    if not response_text:
        return {"action_type": "finish", "email_id": None, "payload": {}}

    text = response_text.strip()

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    candidates = [text]

    depth, start_idx = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(text[start_idx:i + 1])
                break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "action_type" in parsed:
                parsed["payload"] = (
                    parsed.get("payload")
                    if isinstance(parsed.get("payload"), dict)
                    else {}
                )
                eid = parsed.get("email_id")
                parsed["email_id"] = str(eid) if eid and eid != "null" else None
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    print(f"WARNING: Could not parse JSON, using fallback. Raw: {text[:100]}", flush=True)
    return {"action_type": "finish", "email_id": None, "payload": {}}


# ---------------------------------------------------------------------------
# Structured validator printing helpers
# ---------------------------------------------------------------------------

def print_start(task_id: str):
    print(f"[START] task={task_id}", flush=True)

def print_step(task_id: str, step: int, reward: float, done: bool):
    print(
        f"[STEP] task={task_id} step={step} reward={reward:.4f} done={str(done).lower()}",
        flush=True
    )

def print_end(task_id: str, score: float, steps: int):
    print(
        f"[END] task={task_id} score={score:.4f} steps={steps}",
        flush=True
    )


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(client, env, task_id: str) -> dict:
    print_start(task_id)

    try:
        from app.models import Action

        result = env.reset(task_id=task_id)
        obs    = result.observation.model_dump()
        done   = result.done
        total_reward = 0.0
        step   = 0

        while not done and step < MAX_STEPS:
            step += 1
            user_prompt = build_user_prompt(obs)

            raw = '{"action_type": "finish", "email_id": null, "payload": {}}'

            if client is not None and OPENAI_API_KEY:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=0,
                        max_tokens=512,
                    )
                    if response.choices:
                        raw = response.choices[0].message.content or raw
                except Exception as api_err:
                    print(f"API error at step {step}: {api_err}", flush=True)

            action_dict = parse_action(raw)
            action = Action(
                action_type=str(action_dict.get("action_type", "finish")),
                email_id=action_dict.get("email_id"),
                payload=(
                    action_dict.get("payload")
                    if isinstance(action_dict.get("payload"), dict)
                    else {}
                ),
            )

            step_result   = env.step(action)
            obs           = step_result.observation.model_dump()
            reward        = step_result.reward
            done          = step_result.done
            total_reward += reward.score

            print_step(task_id, step, reward.score, done)

        grade = env.grade()
        final_score = float(grade.get("score", 0.0))
        print_end(task_id, final_score, step)
        return grade

    except Exception as task_err:
        print(f"Task {task_id} error: {task_err}", flush=True)
        print_end(task_id, 0.0, 0)
        return {"score": 0.0, "summary": f"Error: {task_err}"}


# ---------------------------------------------------------------------------
# Main - NEVER exits with non-zero code
# ---------------------------------------------------------------------------

def main():
    print("InboxPilot - Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)

    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set - running dry baseline mode.", flush=True)

    try:
        from app.env   import InboxPilotEnv
        from app.tasks import get_all_task_ids
    except Exception as import_err:
        print(f"Failed to import app modules: {import_err}", flush=True)
        return

    try:
        env      = InboxPilotEnv()
        task_ids = get_all_task_ids()
    except Exception as setup_err:
        print(f"Environment setup error: {setup_err}", flush=True)
        return

    client = None
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
        except Exception as client_err:
            print(f"Could not create OpenAI client: {client_err}", flush=True)

    results = {}
    for tid in task_ids:
        try:
            results[tid] = run_task(client, env, tid)
        except Exception as e:
            print(f"Task {tid} failed: {e}", flush=True)
            print_end(tid, 0.0, 0)
            results[tid] = {"score": 0.0, "summary": f"Error: {e}"}

    print("\n=== FINAL SUMMARY ===", flush=True)
    scores = []
    for tid, grade in results.items():
        s = float(grade.get("score", 0.0))
        scores.append(s)
        print(f"{tid}: {s:.4f}", flush=True)
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"AVERAGE: {avg:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
    sys.exit(0)
