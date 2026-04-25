"""Email workflow environment with prioritization and explainable actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from environment.calendar_env import CalendarEnv
from utils.reward import evaluate_action_reward, evaluate_ordering_reward
from utils.scoring import PriorityBreakdown, compute_priority_score


@dataclass
class EmailItem:
    email_id: str
    sender: str
    sender_importance: str
    subject: str
    body: str
    urgency: str
    deadline: datetime | None
    requires_scheduling: bool = False
    preferred_slot: str | None = None


@dataclass
class WorkflowState:
    processed_order: list[str] = field(default_factory=list)
    classifications: dict[str, str] = field(default_factory=dict)
    action_logs: list[dict[str, str]] = field(default_factory=list)
    decision_trace: list[str] = field(default_factory=list)
    reward_total: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)


class EmailEnv:
    """Simple simulation-first environment for decision-making demos."""

    def __init__(self, inbox: list[EmailItem], now: datetime | None = None) -> None:
        self.inbox = list(inbox)
        self.now = now or datetime.now(timezone.utc)
        self.calendar = CalendarEnv(
            time_slots=[
                "tomorrow-09:00",
                "tomorrow-10:00",
                "tomorrow-11:00",
                "tomorrow-14:00",
            ]
        )
        self.state = WorkflowState()

    @staticmethod
    def demo_inbox(now: datetime | None = None) -> list[EmailItem]:
        reference = now or datetime.now(timezone.utc)
        return [
            EmailItem(
                email_id="email_001",
                sender="Boss",
                sender_importance="boss",
                subject="Submit report in 1 hour",
                body="Please send the board report in the next hour.",
                urgency="high",
                deadline=reference + timedelta(hours=1),
                requires_scheduling=False,
            ),
            EmailItem(
                email_id="email_002",
                sender="Client",
                sender_importance="important",
                subject="Schedule meeting tomorrow",
                body="Can we schedule a meeting for tomorrow morning?",
                urgency="medium",
                deadline=reference + timedelta(days=1),
                requires_scheduling=True,
                preferred_slot="tomorrow-10:00",
            ),
            EmailItem(
                email_id="email_003",
                sender="Friend",
                sender_importance="normal",
                subject="Party tonight?",
                body="Hey, are you joining the party tonight?",
                urgency="low",
                deadline=reference + timedelta(hours=8),
                requires_scheduling=False,
            ),
            EmailItem(
                email_id="email_004",
                sender="Unknown Promo",
                sender_importance="spam",
                subject="Win iPhone",
                body="Claim your free iPhone now by clicking this link.",
                urgency="low",
                deadline=None,
                requires_scheduling=False,
            ),
        ]

    def reset(self) -> None:
        self.state = WorkflowState()
        self.calendar.scheduled_tasks.clear()
        self.calendar.rejected_tasks.clear()
        # Pre-book a slot to create a realistic conflict for poor policies.
        self.calendar.schedule_task("tomorrow-10:00", "existing-team-sync")

    def classify_email(self, email: EmailItem) -> str:
        lowered = f"{email.subject} {email.body}".lower()
        if email.sender_importance == "spam" or "win iphone" in lowered:
            return "spam"
        if "meeting" in lowered or email.requires_scheduling:
            return "meeting_request"
        if email.sender_importance == "boss" or "report" in lowered:
            return "urgent_work"
        return "personal"

    def score_all(self) -> dict[str, PriorityBreakdown]:
        scored: dict[str, PriorityBreakdown] = {}
        for email in self.inbox:
            scored[email.email_id] = compute_priority_score(
                urgency=email.urgency,
                sender_importance=email.sender_importance,
                deadline=email.deadline,
                now=self.now,
            )
        return scored

    def run_episode(self, agent: Any) -> dict[str, Any]:
        """Run classify -> prioritize -> action workflow across the inbox."""
        self.reset()

        scored = self.score_all()
        ordered_ids = agent.choose_processing_order(self.inbox, scored)
        email_map = {email.email_id: email for email in self.inbox}

        for email_id in ordered_ids:
            email = email_map[email_id]
            classification = self.classify_email(email)
            self.state.classifications[email.email_id] = classification

            priority = scored[email.email_id]
            decision = agent.decide_action(email, classification, priority, self.calendar)
            action = str(decision.get("action", "ignore"))
            reason = str(decision.get("reason", "No reason provided"))
            reply_text = str(decision.get("reply", ""))

            schedule_conflict = False
            if action == "schedule":
                preferred_slot = str(decision.get("time_slot") or email.preferred_slot or "")
                if preferred_slot:
                    schedule_result = self.calendar.schedule_task(preferred_slot, email.email_id)
                    if not schedule_result["success"] and schedule_result["conflict"]:
                        schedule_conflict = True
                        fallback_slot = self.calendar.first_available_slot()
                        if fallback_slot and bool(decision.get("allow_reschedule", False)):
                            move_result = self.calendar.schedule_task(fallback_slot, email.email_id)
                            schedule_conflict = not bool(move_result.get("success"))
                else:
                    self.calendar.reject_task(email.email_id)

            if action == "reject_task":
                self.calendar.reject_task(email.email_id)
                action = "ignore"

            polite_reply = _is_polite_reply(reply_text)
            is_high_priority = priority.priority_score >= 8
            is_spam = email.sender_importance == "spam" or classification == "spam"

            reward_event = evaluate_action_reward(
                action=action,
                is_high_priority=is_high_priority,
                is_spam=is_spam,
                scheduling_conflict=schedule_conflict,
                polite_reply=polite_reply,
            )
            self._apply_reward_event(reward_event)

            self.state.processed_order.append(email.email_id)
            self.state.action_logs.append(
                {
                    "email_id": email.email_id,
                    "action": action,
                    "reason": reason,
                }
            )
            self.state.decision_trace.append(
                " -> ".join(
                    [
                        f"email={email.email_id}",
                        f"classify={classification}",
                        f"priority={priority.priority_score}",
                        f"action={action}",
                    ]
                )
            )

        ideal_order = [
            item[0]
            for item in sorted(
                scored.items(),
                key=lambda item: item[1].priority_score,
                reverse=True,
            )
        ]
        first_id = self.state.processed_order[0] if self.state.processed_order else ""
        first_is_urgent = bool(first_id and scored[first_id].priority_score >= 8)

        ordering_event = evaluate_ordering_reward(
            processed_order=self.state.processed_order,
            ideal_order=ideal_order,
            first_email_is_urgent=first_is_urgent,
        )
        self._apply_reward_event(ordering_event)

        return {
            "agent": agent.name,
            "processed_order": self.state.processed_order,
            "priorities": {
                eid: {
                    "score": breakdown.priority_score,
                    "reason": breakdown.reason,
                }
                for eid, breakdown in scored.items()
            },
            "decision_trace": self.state.decision_trace,
            "action_logs": self.state.action_logs,
            "reward_total": round(self.state.reward_total, 2),
            "reward_breakdown": self.state.reward_breakdown,
            "calendar": {
                "scheduled": dict(self.calendar.scheduled_tasks),
                "rejected": list(self.calendar.rejected_tasks),
            },
        }

    def _apply_reward_event(self, event: Any) -> None:
        self.state.reward_total += float(event.score)
        for key, value in event.breakdown.items():
            self.state.reward_breakdown[key] = round(
                self.state.reward_breakdown.get(key, 0.0) + float(value),
                2,
            )


def _is_polite_reply(reply_text: str) -> bool:
    text = reply_text.strip().lower()
    if not text:
        return False
    polite_tokens = ["please", "thanks", "thank you", "happy to", "best"]
    return any(token in text for token in polite_tokens)
