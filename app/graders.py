"""
InboxPilot — Deterministic graders.

Each grader evaluates an episode's final state against the task answer keys.
All scores are in [0.0, 1.0]. NO LLM-based or fuzzy evaluation — only
exact-match and controlled keyword checks.
"""

from __future__ import annotations

from .models import AnswerKey, EnvironmentState, TaskDefinition
from .utils import clamp, keyword_match_ratio, normalize


def _per_email_score(ak: AnswerKey, state: EnvironmentState) -> dict[str, float]:
    """Score a single email against its answer key. Returns component scores."""
    eid = ak.email_id
    scores: dict[str, float] = {}

    # Classification accuracy (0 or 1)
    actual_class = normalize(state.classifications.get(eid, ""))
    expected_class = normalize(ak.classification)
    scores["classification"] = 1.0 if actual_class == expected_class else 0.0

    # Priority correctness (0 or 1)
    actual_prio = normalize(state.priorities.get(eid, ""))
    expected_prio = normalize(ak.priority)
    scores["priority"] = 1.0 if actual_prio == expected_prio else 0.0

    # Correct action taken
    expected_action = ak.action
    action_taken = False
    if expected_action == "escalate":
        action_taken = eid in state.escalations
    elif expected_action == "mark_spam":
        action_taken = eid in state.spam_flags
    elif expected_action == "archive":
        action_taken = eid in state.archives
    elif expected_action == "send_reply":
        action_taken = eid in state.sent_replies
    scores["action"] = 1.0 if action_taken else 0.0

    # Escalation target correctness (if applicable)
    if ak.escalation_target:
        actual_target = normalize(state.escalations.get(eid, ""))
        expected_target = normalize(ak.escalation_target)
        scores["escalation_target"] = 1.0 if actual_target == expected_target else 0.0

    # Reply keyword coverage (if applicable)
    if ak.reply_required_keywords:
        reply_text = state.sent_replies.get(eid, state.drafts.get(eid, ""))
        scores["reply_quality"] = keyword_match_ratio(reply_text, ak.reply_required_keywords)

    # Follow-up scheduled (if required)
    if ak.followup_required:
        scores["followup"] = 1.0 if eid in state.followups else 0.0

    # Unsafe handling check — did agent reply to spam/phishing?
    if ak.action == "mark_spam" and eid in state.sent_replies:
        scores["unsafe_reply_to_spam"] = 0.0  # penalty via lower score
        scores["action"] = 0.0  # override — wrong action

    return scores


def grade_task(state: EnvironmentState, task: TaskDefinition) -> dict:
    """
    Grade the final state of an episode for a given task.

    Returns:
        {
            "task_id": str,
            "score": float,          # overall score in [0.0, 1.0]
            "per_email": {...},       # per-email breakdown
            "efficiency": float,      # step efficiency score
            "summary": str,
        }
    """
    try:
        return _grade_task_inner(state, task)
    except Exception as e:
        return {
            "task_id": getattr(task, "task_id", "unknown"),
            "score": 0.0,
            "correctness": 0.0,
            "efficiency": 0.0,
            "per_email": {},
            "summary": f"Grading error (safe fallback): {e}",
        }


def _grade_task_inner(state: EnvironmentState, task: TaskDefinition) -> dict:
    """Inner grading logic — separated for safe wrapping."""
    if not task.answer_keys:
        return {"task_id": task.task_id, "score": 0.0, "per_email": {}, "efficiency": 0.0, "summary": "No answer keys."}

    per_email: dict[str, dict] = {}
    all_component_scores: list[float] = []

    for ak in task.answer_keys:
        email_scores = _per_email_score(ak, state)
        avg = clamp(sum(email_scores.values()) / len(email_scores)) if email_scores else 0.0
        per_email[ak.email_id] = {
            "components": email_scores,
            "average": round(avg, 4),
        }
        all_component_scores.append(avg)

    # Average across all emails
    correctness = clamp(sum(all_component_scores) / len(all_component_scores)) if all_component_scores else 0.0

    # Efficiency bonus: using fewer steps is better
    ideal_steps = max(1, len(task.answer_keys) * 4)  # ~4 actions per email (open, classify, priority, action)
    actual_steps = max(0, state.step_count)
    max_steps = max(ideal_steps + 1, task.max_steps)  # prevent division by zero
    if actual_steps <= ideal_steps:
        efficiency = 1.0
    elif actual_steps <= max_steps:
        efficiency = clamp(1.0 - (actual_steps - ideal_steps) / (max_steps - ideal_steps + 1))
    else:
        efficiency = 0.0

    # Final score: 85% correctness, 15% efficiency
    final_score = clamp(0.85 * correctness + 0.15 * efficiency)

    summary_parts = []
    summary_parts.append(f"Correctness: {correctness:.2%}")
    summary_parts.append(f"Efficiency: {efficiency:.2%}")
    summary_parts.append(f"Steps used: {actual_steps}/{task.max_steps}")
    summary_parts.append(f"Final score: {final_score:.4f}")

    return {
        "task_id": task.task_id,
        "score": round(final_score, 4),
        "correctness": round(correctness, 4),
        "efficiency": round(efficiency, 4),
        "per_email": per_email,
        "summary": " | ".join(summary_parts),
    }
