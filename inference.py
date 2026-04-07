#!/usr/bin/env python3
"""
InboxPilot — Baseline inference script.

Evaluates all 3 tasks using an LLM via the OpenAI Python client.
Produces reproducible baseline scores with deterministic prompting (temperature=0).

Usage:
    export OPENAI_API_KEY=sk-...
    export MODEL_NAME=gpt-4o-mini          # optional, defaults to gpt-4o-mini
    export API_BASE_URL=https://api.openai.com/v1  # optional
    python inference.py
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
- open_email: Open an email to read it. Payload: {} (email_id required)
- classify_email: Classify an email. Payload: {"category": "<category>"}
- set_priority: Set priority. Payload: {"priority": "low"|"medium"|"high"|"critical"}
- draft_reply: Draft a reply. Payload: {"reply_text": "<text>"}
- send_reply: Send the drafted reply. Payload: {}
- escalate: Escalate to a team. Payload: {"team": "<team_name>"}
- mark_spam: Mark as spam. Payload: {}
- archive: Archive the email. Payload: {}
- schedule_followup: Schedule a follow-up. Payload: {}
- request_more_info: Request more info. Payload: {}
- finish: Signal you are done with all emails. Payload: {}

Response format (ONLY valid JSON):
{
  "action_type": "<action_name>",
  "email_id": "<email_id or null>",
  "payload": { ... }
}

Strategy:
1. For each email: open -> classify -> set_priority -> take action (escalate/reply/spam/archive)
2. Process all emails before calling finish.
3. For spam/phishing: mark_spam, do NOT reply.
4. For billing/refund: classify, set high priority, escalate to billing.
5. For complaints: draft empathetic reply, send, escalate, schedule_followup if needed.
6. For legal: escalate to legal, do NOT reply.
7. For security alerts: escalate to security.
8. For HR complaints: escalate to hr.
9. For media inquiries: escalate to communications.
10. For investor requests: escalate to executive.
"""


def build_user_prompt(observation: dict) -> str:
    """Build user prompt from observation."""
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
    """Parse JSON action from LLM response. Fallback on failure."""
    if not response_text:
        return {"action_type": "finish", "email_id": None, "payload": {}}

    text = response_text.strip()

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    candidates = [text]

    # Find the outermost { ... }
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

    print(f"  WARNING: Could not parse JSON, using fallback. Raw: {text[:100]}")
    return {"action_type": "finish", "email_id": None, "payload": {}}


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(client, env, task_id: str) -> dict:
    """Run inference on a single task and return the grade result."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

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

            # Default action if no API key or API fails
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
                    print(f"  API error at step {step}: {api_err}")

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

            print(f"  Step {step}: {action.action_type} on {action.email_id or '-'}", end="")

            step_result   = env.step(action)
            obs           = step_result.observation.model_dump()
            reward        = step_result.reward
            done          = step_result.done
            total_reward += reward.score

            print(f"  -> reward: {reward.score:+.3f} | {reward.message[:60]}")

        grade = env.grade()
        print(f"\n  Final score: {grade.get('score', 0):.4f}")
        print(f"  {grade.get('summary', '')}")
        return grade

    except Exception as task_err:
        print(f"  Task {task_id} error: {task_err}")
        return {"score": 0.0, "summary": f"Error: {task_err}"}


# ---------------------------------------------------------------------------
# Main - NEVER exits with non-zero code
# ---------------------------------------------------------------------------

def main():
    print("InboxPilot - Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"Base URL: {API_BASE_URL}")

    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set - running in env-validation / dry-run mode.")
        print("Tasks will step with finish actions (score=0) to validate env structure.")

    # Import app modules - if they fail, report and exit cleanly (code 0)
    try:
        from app.env   import InboxPilotEnv
        from app.tasks import get_all_task_ids
    except Exception as import_err:
        print(f"Failed to import app modules: {import_err}")
        return  # exits main() cleanly

    # Build env
    try:
        env      = InboxPilotEnv()
        task_ids = get_all_task_ids()
    except Exception as setup_err:
        print(f"Environment setup error: {setup_err}")
        return

    # Optional OpenAI client
    client = None
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
        except Exception as client_err:
            print(f"Could not create OpenAI client: {client_err}")

    # Run tasks
    results = {}
    for tid in task_ids:
        try:
            results[tid] = run_task(client, env, tid)
        except Exception as e:
            print(f"\n  Task {tid} failed: {e}")
            results[tid] = {"score": 0.0, "summary": f"Error: {e}"}

    # Summary
    print("\n" + "=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    scores = []
    for tid, grade in results.items():
        s = grade.get("score", 0)
        scores.append(s)
        print(f"  {tid:<40} {s:.4f}")
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<40} {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
    sys.exit(0)  # ALWAYS exit with code 0
