from __future__ import annotations

from typing import Any

from inboxpilot.environment import classify_email, priority_score, sorted_emails_by_priority


class UntrainedAgent:
    name = "untrained_agent"

    def run(self, emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
        order = ["email_003", "email_002", "email_001", "email_004"]
        by_id = {email["email_id"]: email for email in emails}

        actions: list[dict[str, Any]] = []
        for email_id in order:
            email = by_id[email_id]
            cls = classify_email(email)
            score, reason = priority_score(email)

            if cls == "meeting_request":
                action = "schedule"
                reply = ""
            elif cls == "spam":
                action = "reply"
                reply = "Clicking now"
            elif email_id == "email_003":
                action = "ignore"
                reply = ""
            else:
                action = "reply"
                reply = "OK"

            actions.append(
                {
                    "email_id": email_id,
                    "classification": cls,
                    "priority_score": score,
                    "action": action,
                    "reply": reply,
                    "reason": reason,
                }
            )

        return actions


class TrainedAgent:
    name = "trained_agent"

    def run(self, emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted_emails_by_priority(emails)
        actions: list[dict[str, Any]] = []

        for email in ranked:
            email_id = email["email_id"]
            cls = classify_email(email)
            score, reason = priority_score(email)

            if cls == "spam":
                action = "ignore"
                reply = ""
            elif cls == "meeting_request":
                action = "schedule"
                reply = ""
            elif email_id == "email_001":
                action = "reply"
                reply = "Thanks. I am finalizing the report now and will submit it within the hour."
            else:
                action = "reply"
                reply = "Thanks for the invite. Happy to join tonight."

            actions.append(
                {
                    "email_id": email_id,
                    "classification": cls,
                    "priority_score": score,
                    "action": action,
                    "reply": reply,
                    "reason": reason,
                }
            )

        return actions
