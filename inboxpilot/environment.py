from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


URGENCY_WEIGHTS = {"low": 1, "medium": 3, "high": 5}
SENDER_WEIGHTS = {"spam": -3, "normal": 1, "important": 3, "boss": 5}


@dataclass
class InboxPilotEnvironment:
    """Lightweight OpenEnv-style simulation for CPU demos."""

    emails: list[dict[str, Any]]
    max_steps: int = 20
    step_count: int = 0
    done: bool = False
    action_log: list[dict[str, str]] = field(default_factory=list)

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self.done = False
        self.action_log = []
        return self.state()

    def step(self, action: dict[str, str]) -> dict[str, Any]:
        if self.done:
            return {"observation": self.state(), "done": True}

        self.step_count += 1
        self.action_log.append(action)
        if self.step_count >= self.max_steps:
            self.done = True

        return {
            "observation": self.state(),
            "done": self.done,
        }

    def state(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "emails": self.emails,
            "action_log": self.action_log,
        }


def _deadline_weight(deadline_iso: str | None, now: datetime) -> int:
    if not deadline_iso:
        return 0

    try:
        deadline = datetime.fromisoformat(deadline_iso)
    except ValueError:
        return 0

    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    minutes = (deadline - now).total_seconds() / 60.0
    if minutes <= 0:
        return 5
    if minutes <= 60:
        return 4
    if minutes <= 24 * 60:
        return 2
    return 1


def classify_email(email: dict[str, Any]) -> str:
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
    if email.get("sender_importance") == "spam" or "win iphone" in text:
        return "spam"
    if "meeting" in text:
        return "meeting_request"
    if email.get("sender_importance") == "boss" or "report" in text:
        return "urgent_work"
    return "personal"


def priority_score(email: dict[str, Any], now: datetime | None = None) -> tuple[int, str]:
    current = now or datetime.now(timezone.utc)
    urgency = str(email.get("urgency", "low")).lower()
    sender_importance = str(email.get("sender_importance", "normal")).lower()

    urgency_weight = URGENCY_WEIGHTS.get(urgency, 1)
    sender_weight = SENDER_WEIGHTS.get(sender_importance, 1)
    deadline_weight = _deadline_weight(email.get("deadline"), current)
    score = urgency_weight + sender_weight + deadline_weight

    if deadline_weight >= 4:
        deadline_text = "near deadline"
    elif deadline_weight > 0:
        deadline_text = "upcoming deadline"
    else:
        deadline_text = "no deadline"

    reason = f"Chosen because: {urgency} urgency + {sender_importance} sender + {deadline_text}"
    return score, reason


def sorted_emails_by_priority(emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[tuple[dict[str, Any], int]] = []
    now = datetime.now(timezone.utc)
    for email in emails:
        score, _ = priority_score(email, now=now)
        enriched.append((email, score))

    return [item[0] for item in sorted(enriched, key=lambda x: x[1], reverse=True)]
