"""Priority scoring helpers for email triage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


URGENCY_WEIGHTS = {
    "low": 1,
    "medium": 3,
    "high": 5,
}

SENDER_WEIGHTS = {
    "spam": -3,
    "normal": 1,
    "important": 3,
    "boss": 5,
}


@dataclass(frozen=True)
class PriorityBreakdown:
    urgency_weight: int
    sender_weight: int
    deadline_weight: int
    priority_score: int
    reason: str


def _deadline_weight(deadline: Optional[datetime], now: datetime) -> int:
    if deadline is None:
        return 0

    delta_minutes = (deadline - now).total_seconds() / 60.0
    if delta_minutes <= 0:
        return 5
    if delta_minutes <= 60:
        return 4
    if delta_minutes <= 6 * 60:
        return 3
    if delta_minutes <= 24 * 60:
        return 2
    if delta_minutes <= 72 * 60:
        return 1
    return 0


def compute_priority_score(
    urgency: str,
    sender_importance: str,
    deadline: Optional[datetime],
    now: Optional[datetime] = None,
) -> PriorityBreakdown:
    """Compute triage priority score and human-readable reasoning text."""
    reference_time = now or datetime.now(timezone.utc)

    urgency_key = (urgency or "low").strip().lower()
    sender_key = (sender_importance or "normal").strip().lower()

    urgency_weight = URGENCY_WEIGHTS.get(urgency_key, URGENCY_WEIGHTS["low"])
    sender_weight = SENDER_WEIGHTS.get(sender_key, SENDER_WEIGHTS["normal"])
    deadline_weight = _deadline_weight(deadline, reference_time)

    score = urgency_weight + sender_weight + deadline_weight

    deadline_phrase = "no deadline"
    if deadline_weight >= 4:
        deadline_phrase = "near deadline"
    elif deadline_weight > 0:
        deadline_phrase = "upcoming deadline"

    reason = (
        f"Chosen because: {urgency_key} urgency + "
        f"{sender_key} sender + {deadline_phrase}"
    )

    return PriorityBreakdown(
        urgency_weight=urgency_weight,
        sender_weight=sender_weight,
        deadline_weight=deadline_weight,
        priority_score=score,
        reason=reason,
    )
