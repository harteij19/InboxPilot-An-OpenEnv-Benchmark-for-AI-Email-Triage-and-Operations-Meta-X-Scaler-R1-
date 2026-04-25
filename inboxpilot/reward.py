from __future__ import annotations

from typing import Any


def compute_total_reward(actions: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    score = 0.0
    breakdown: dict[str, float] = {
        "urgent_first": 0.0,
        "spam_handling": 0.0,
        "scheduling": 0.0,
        "reply_tone": 0.0,
    }

    if actions:
        first = actions[0]
        if first.get("email_id") == "email_001":
            breakdown["urgent_first"] += 2.0
        else:
            breakdown["urgent_first"] -= 1.0

    for action in actions:
        email_id = action.get("email_id")
        action_type = action.get("action")
        reply = str(action.get("reply", "")).lower()

        if email_id == "email_004":
            if action_type == "ignore":
                breakdown["spam_handling"] += 2.0
            else:
                breakdown["spam_handling"] -= 2.0

        if email_id == "email_002":
            if action_type == "schedule":
                breakdown["scheduling"] += 1.5
            else:
                breakdown["scheduling"] -= 1.0

        if action_type == "reply":
            if any(token in reply for token in ["thanks", "thank you", "please", "happy"]):
                breakdown["reply_tone"] += 0.75
            else:
                breakdown["reply_tone"] -= 0.5

    score = round(sum(breakdown.values()), 2)
    return score, breakdown
