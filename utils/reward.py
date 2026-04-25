"""Reward utilities for multi-step email decision workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardEvent:
    score: float
    breakdown: dict[str, float]
    message: str


def evaluate_action_reward(
    *,
    action: str,
    is_high_priority: bool,
    is_spam: bool,
    scheduling_conflict: bool,
    polite_reply: bool,
) -> RewardEvent:
    """Return a reward event aligned with demo-focused behavior incentives."""
    score = 0.0
    breakdown: dict[str, float] = {}
    messages: list[str] = []

    if action == "ignore" and is_high_priority:
        breakdown["ignored_high_priority"] = -2.0
        score -= 2.0
        messages.append("Ignored a high-priority email.")

    if action == "schedule" and not scheduling_conflict:
        breakdown["successful_scheduling"] = 1.25
        score += 1.25
        messages.append("Scheduled without conflicts.")

    if action == "schedule" and scheduling_conflict:
        breakdown["schedule_conflict"] = -1.5
        score -= 1.5
        messages.append("Scheduling conflict detected.")

    if action == "reply" and polite_reply:
        breakdown["polite_reply"] = 0.75
        score += 0.75
        messages.append("Reply is polite and context-aware.")

    if action == "reply" and not polite_reply:
        breakdown["poor_tone"] = -1.0
        score -= 1.0
        messages.append("Reply tone is poor.")

    if is_spam and action != "ignore":
        breakdown["acted_on_spam"] = -2.0
        score -= 2.0
        messages.append("Agent acted on spam.")

    return RewardEvent(score=score, breakdown=breakdown, message=" ".join(messages))


def evaluate_ordering_reward(
    processed_order: list[str],
    ideal_order: list[str],
    first_email_is_urgent: bool,
) -> RewardEvent:
    """Reward prioritization quality and urgent-first execution."""
    score = 0.0
    breakdown: dict[str, float] = {}
    messages: list[str] = []

    if processed_order == ideal_order:
        breakdown["correct_prioritization"] = 2.0
        score += 2.0
        messages.append("Emails processed in optimal priority order.")
    else:
        matched_positions = sum(
            1 for idx, email_id in enumerate(processed_order)
            if idx < len(ideal_order) and ideal_order[idx] == email_id
        )
        positional_score = round(0.4 * matched_positions, 2)
        if positional_score > 0:
            breakdown["partial_prioritization"] = positional_score
            score += positional_score
            messages.append("Partially matched expected prioritization.")

    if first_email_is_urgent:
        breakdown["urgent_first"] = 1.0
        score += 1.0
        messages.append("Handled urgent email first.")
    else:
        breakdown["urgent_first_missed"] = -1.0
        score -= 1.0
        messages.append("Did not handle urgent email first.")

    return RewardEvent(score=score, breakdown=breakdown, message=" ".join(messages))
