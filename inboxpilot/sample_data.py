from __future__ import annotations

from datetime import datetime, timedelta, timezone


def get_sample_emails() -> list[dict[str, str | None]]:
    now = datetime.now(timezone.utc)
    return [
        {
            "email_id": "email_001",
            "sender": "Boss",
            "sender_importance": "boss",
            "subject": "Submit report in 1 hour",
            "body": "Please send the board report in the next hour.",
            "urgency": "high",
            "deadline": (now + timedelta(hours=1)).isoformat(),
        },
        {
            "email_id": "email_002",
            "sender": "Client",
            "sender_importance": "important",
            "subject": "Schedule meeting tomorrow",
            "body": "Can we schedule a meeting tomorrow morning?",
            "urgency": "medium",
            "deadline": (now + timedelta(days=1)).isoformat(),
        },
        {
            "email_id": "email_003",
            "sender": "Friend",
            "sender_importance": "normal",
            "subject": "Party tonight?",
            "body": "Hey, are you joining tonight?",
            "urgency": "low",
            "deadline": None,
        },
        {
            "email_id": "email_004",
            "sender": "Promo",
            "sender_importance": "spam",
            "subject": "Win iPhone",
            "body": "Claim your free iPhone now.",
            "urgency": "low",
            "deadline": None,
        },
    ]
