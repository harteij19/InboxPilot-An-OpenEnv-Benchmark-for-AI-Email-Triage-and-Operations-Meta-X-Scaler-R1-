"""
InboxPilot — Reward engine.

Computes per-step shaped rewards based on action correctness,
using the answer keys from the current task definition.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import Action, AnswerKey, EnvironmentState, Reward, TaskDefinition
from .utils import clamp, keyword_match_ratio, normalize


def _find_answer_key(task: TaskDefinition, email_id: str) -> Optional[AnswerKey]:
    """Find the answer key for a given email_id."""
    for ak in task.answer_keys:
        if ak.email_id == email_id:
            return ak
    return None


def _count_action_repeats(state: EnvironmentState, action_type: str, email_id: Optional[str]) -> int:
    """Count how many times this exact action was already taken."""
    count = 0
    for h in state.action_history:
        if h.get("action_type") == action_type and h.get("email_id") == email_id:
            count += 1
    return count


def compute_reward(
    action: Action,
    state: EnvironmentState,
    task: TaskDefinition,
) -> Reward:
    """Compute a shaped reward for the given action in the current state."""
    try:
        return _compute_reward_inner(action, state, task)
    except Exception as e:
        return Reward(
            score=-0.1,
            breakdown={"error": -0.1},
            message=f"Reward computation error (safe fallback): {e}",
        )


def _compute_reward_inner(
    action: Action,
    state: EnvironmentState,
    task: TaskDefinition,
) -> Reward:
    """Inner reward computation — may raise on truly malformed input."""
    breakdown: dict[str, float] = {}
    messages: list[str] = []
    score = 0.0

    at = action.action_type
    eid = action.email_id
    payload = action.payload

    # --- Penalty: repeated / looping actions ---
    repeats = _count_action_repeats(state, at, eid)
    if repeats >= 2:
        breakdown["repeat_penalty"] = -0.1
        score -= 0.1
        messages.append("Repeated action detected.")

    # --- Penalty: action on nonexistent email ---
    valid_ids = {e.email_id for e in state.emails}
    if eid and eid not in valid_ids:
        breakdown["invalid_email"] = -0.2
        score -= 0.2
        messages.append(f"Email '{eid}' does not exist.")
        return Reward(score=score, breakdown=breakdown, message=" ".join(messages))

    # --- Penalty: acting without opening the email first ---
    needs_open = at in {
        "classify_email", "set_priority", "draft_reply", "send_reply",
        "escalate", "mark_spam", "archive", "schedule_followup", "request_more_info",
    }
    opened_ids = {
        h["email_id"] for h in state.action_history
        if h.get("action_type") == "open_email" and h.get("email_id")
    }
    if needs_open and eid and eid not in opened_ids:
        breakdown["not_opened_penalty"] = -0.05
        score -= 0.05
        messages.append("Email not opened before acting on it.")

    ak = _find_answer_key(task, eid) if eid else None

    # --- Action-specific rewards ---
    if at == "open_email":
        if eid and eid in valid_ids:
            breakdown["open"] = 0.02
            score += 0.02
            messages.append("Email opened.")

    elif at == "classify_email":
        category = normalize(payload.get("category", ""))
        if ak:
            if category == normalize(ak.classification):
                breakdown["classification"] = 0.15
                score += 0.15
                messages.append("Correct classification.")
            else:
                breakdown["classification"] = -0.05
                score -= 0.05
                messages.append(f"Wrong classification: '{category}' vs expected '{ak.classification}'.")

    elif at == "set_priority":
        priority = normalize(payload.get("priority", ""))
        if ak:
            if priority == normalize(ak.priority):
                breakdown["priority"] = 0.10
                score += 0.10
                messages.append("Correct priority.")
            else:
                breakdown["priority"] = -0.05
                score -= 0.05
                messages.append(f"Wrong priority: '{priority}' vs expected '{ak.priority}'.")

    elif at == "draft_reply":
        reply_text = payload.get("reply_text", "")
        if ak and ak.reply_required_keywords:
            ratio = keyword_match_ratio(reply_text, ak.reply_required_keywords)
            r = round(0.10 * ratio, 4)
            breakdown["draft_quality"] = r
            score += r
            messages.append(f"Draft quality: {ratio:.0%} keywords matched.")
        elif reply_text:
            breakdown["draft"] = 0.02
            score += 0.02
            messages.append("Draft created.")

    elif at == "send_reply":
        # Check if a draft exists first
        if eid and eid in state.drafts:
            reply_text = state.drafts[eid]
            if ak and ak.reply_required_keywords:
                ratio = keyword_match_ratio(reply_text, ak.reply_required_keywords)
                r = round(0.10 * ratio, 4)
                breakdown["reply_quality"] = r
                score += r
                messages.append(f"Reply sent — keyword coverage: {ratio:.0%}.")
            if ak and ak.action == "send_reply":
                breakdown["correct_action"] = 0.05
                score += 0.05
                messages.append("Correct action: send reply.")
        elif eid and eid not in state.drafts:
            breakdown["no_draft_penalty"] = -0.05
            score -= 0.05
            messages.append("Sent reply without a draft.")

    elif at == "escalate":
        team = normalize(payload.get("team", ""))
        if ak:
            if ak.action == "escalate":
                breakdown["correct_action"] = 0.05
                score += 0.05
                messages.append("Correct action: escalate.")
            if ak.escalation_target and team == normalize(ak.escalation_target):
                breakdown["escalation_target"] = 0.15
                score += 0.15
                messages.append("Correct escalation target.")
            elif ak.escalation_target:
                breakdown["escalation_target"] = -0.05
                score -= 0.05
                messages.append(f"Wrong escalation target: '{team}' vs expected '{ak.escalation_target}'.")

    elif at == "mark_spam":
        if ak and ak.action == "mark_spam":
            breakdown["correct_spam"] = 0.15
            score += 0.15
            messages.append("Correctly marked as spam.")
        elif ak and ak.action != "mark_spam":
            breakdown["wrong_spam"] = -0.15
            score -= 0.15
            messages.append("Incorrectly marked as spam — this is a legitimate email.")

    elif at == "archive":
        if ak and ak.action == "archive":
            breakdown["correct_archive"] = 0.10
            score += 0.10
            messages.append("Correctly archived.")
        elif eid:
            breakdown["archive"] = 0.01
            score += 0.01
            messages.append("Email archived.")

    elif at == "schedule_followup":
        if ak and ak.followup_required:
            breakdown["followup"] = 0.10
            score += 0.10
            messages.append("Follow-up correctly scheduled.")
        else:
            breakdown["followup"] = 0.01
            score += 0.01
            messages.append("Follow-up scheduled.")

    elif at == "request_more_info":
        breakdown["info_request"] = 0.01
        score += 0.01
        messages.append("Requested more info.")

    elif at == "finish":
        # Reward/penalty for finish handled separately based on completeness
        total_emails = len(task.answer_keys)
        handled = 0
        for a in task.answer_keys:
            eid_check = a.email_id
            has_class = eid_check in state.classifications
            has_prio = eid_check in state.priorities
            has_action = (
                eid_check in state.sent_replies
                or eid_check in state.escalations
                or eid_check in state.spam_flags
                or eid_check in state.archives
            )
            if has_class and has_prio and has_action:
                handled += 1
        completeness = handled / total_emails if total_emails > 0 else 0.0
        if completeness >= 0.8:
            breakdown["finish_bonus"] = 0.15
            score += 0.15
            messages.append(f"Good finish — {completeness:.0%} emails fully handled.")
        elif completeness >= 0.5:
            breakdown["finish_partial"] = 0.05
            score += 0.05
            messages.append(f"Partial finish — only {completeness:.0%} emails handled.")
        else:
            breakdown["premature_finish"] = -0.15
            score -= 0.15
            messages.append(f"Premature finish — only {completeness:.0%} emails handled.")
    else:
        breakdown["invalid_action_type"] = -0.1
        score -= 0.1
        messages.append(f"Unknown action type: '{at}'.")

    # --- Penalty: too many steps ---
    if state.step_count > state.max_steps * 0.8:
        breakdown["step_warning"] = -0.02
        score -= 0.02
        messages.append("Approaching step limit.")

    return Reward(
        score=round(clamp(score, -1.0, 1.0), 4),
        breakdown=breakdown,
        message=" ".join(messages),
    )
