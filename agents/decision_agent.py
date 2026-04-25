"""Baseline and trained decision agents for InboxPilot demos."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.calendar_env import CalendarEnv
    from environment.email_env import EmailItem
    from utils.scoring import PriorityBreakdown


@dataclass
class BaselineAgent:
    """Simple, intentionally weak policy for before/after demos."""

    name: str = "baseline_agent"
    seed: int = 7

    def choose_processing_order(
        self,
        emails: list[EmailItem],
        scored: dict[str, PriorityBreakdown],
    ) -> list[str]:
        _ = scored
        rng = random.Random(self.seed)
        ordered = [email.email_id for email in emails]
        if len(ordered) >= 3:
            # Keeps behavior deterministic but usually suboptimal.
            ordered[0], ordered[2] = ordered[2], ordered[0]
        rng.shuffle(ordered[1:])
        return ordered

    def decide_action(
        self,
        email: EmailItem,
        classification: str,
        priority: PriorityBreakdown,
        calendar: CalendarEnv,
    ) -> dict[str, object]:
        _ = priority
        _ = calendar

        if classification == "meeting_request":
            return {
                "action": "schedule",
                "time_slot": email.preferred_slot or "tomorrow-10:00",
                "allow_reschedule": False,
                "reason": "Baseline policy schedules directly without conflict checks.",
            }

        if classification == "spam":
            return {
                "action": "reply",
                "reply": "Clicking now. Thanks.",
                "reason": "Baseline policy failed to filter spam correctly.",
            }

        if email.sender.lower() == "friend":
            return {
                "action": "ignore",
                "reason": "Baseline policy ignores casual messages.",
            }

        return {
            "action": "reply",
            "reply": "OK",
            "reason": "Baseline generic reply policy.",
        }


@dataclass
class TrainedAgent:
    """Priority-aware policy with explainable actions."""

    name: str = "trained_agent"

    def choose_processing_order(
        self,
        emails: list[EmailItem],
        scored: dict[str, PriorityBreakdown],
    ) -> list[str]:
        _ = emails
        ranked = sorted(
            scored.items(),
            key=lambda item: item[1].priority_score,
            reverse=True,
        )
        return [email_id for email_id, _ in ranked]

    def decide_action(
        self,
        email: EmailItem,
        classification: str,
        priority: PriorityBreakdown,
        calendar: CalendarEnv,
    ) -> dict[str, object]:
        if classification == "spam":
            return {
                "action": "ignore",
                "reason": (
                    "Chosen because: spam signal + low business value + "
                    "risk of unsafe interaction"
                ),
            }

        if classification == "meeting_request" or email.requires_scheduling:
            preferred = email.preferred_slot or "tomorrow-10:00"
            selected_slot = preferred
            if preferred in calendar.scheduled_tasks:
                selected_slot = calendar.first_available_slot() or preferred

            return {
                "action": "schedule",
                "time_slot": selected_slot,
                "allow_reschedule": True,
                "reason": (
                    f"{priority.reason} + meeting intent detected + "
                    f"selected open slot {selected_slot}"
                ),
            }

        if email.sender_importance == "boss":
            return {
                "action": "reply",
                "reply": (
                    "Thanks for the reminder. I am finalizing the report now "
                    "and will submit it within the hour."
                ),
                "reason": f"{priority.reason} + executive sender requires immediate response",
            }

        return {
            "action": "reply",
            "reply": "Thanks for the invite. Happy to join tonight.",
            "reason": f"{priority.reason} + personal context handled politely",
        }
