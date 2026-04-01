"""
InboxPilot — Core environment.

Implements the OpenEnv-compatible InboxPilotEnv class with:
  - reset(task_id) → StepResult
  - step(action)   → StepResult
  - state()        → EnvironmentState
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from .graders import grade_task
from .models import (
    Action,
    AnswerKey,
    Email,
    EmailSummary,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
    TaskDefinition,
)
from .rewards import compute_reward
from .tasks import get_task


class InboxPilotEnv:
    """OpenEnv-compatible email triage environment."""

    def __init__(self) -> None:
        self._state = EnvironmentState()
        self._task: Optional[TaskDefinition] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy_support_triage") -> StepResult:
        """Reset the environment to the beginning of a task."""
        self._task = get_task(task_id)
        self._state = EnvironmentState(
            task_id=task_id,
            emails=copy.deepcopy(self._task.emails),
            max_steps=self._task.max_steps,
        )
        return StepResult(
            observation=self._build_observation(),
            reward=Reward(score=0.0, breakdown={}, message="Episode started."),
            done=False,
            truncated=False,
            info={"task_id": task_id, "num_emails": len(self._task.emails)},
        )

    def step(self, action: Action) -> StepResult:
        """Execute an action and return the result."""
        if self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(score=0.0, breakdown={}, message="Episode already done."),
                done=True,
                truncated=False,
                info=self._build_info(),
            )

        # Record action in history
        self._state.step_count += 1
        action_record = {
            "step": self._state.step_count,
            "action_type": action.action_type,
            "email_id": action.email_id,
            "payload": action.payload if isinstance(action.payload, dict) else {},
        }

        # Compute reward BEFORE applying (reward looks at current state)
        try:
            reward = compute_reward(action, self._state, self._task)
        except Exception:
            reward = Reward(score=-0.1, breakdown={"error": -0.1}, message="Reward computation failed.")

        # Apply action to state
        try:
            self._apply_action(action)
        except Exception:
            pass  # action application failure is non-fatal; state stays unchanged

        # Record in history AFTER applying
        self._state.action_history.append(action_record)
        self._state.total_reward += reward.score

        # Check termination
        terminated = action.action_type == "finish"
        truncated = (not terminated) and (self._state.step_count >= self._state.max_steps)
        if terminated or truncated:
            self._state.done = True
            try:
                grade_result = grade_task(self._state, self._task)
            except Exception:
                grade_result = {"score": 0.0, "summary": "Grading failed."}
            self._state.success = grade_result.get("score", 0.0) >= 0.5
            info = self._build_info()
            info["grade"] = grade_result
        else:
            truncated = False
            info = self._build_info()

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._state.done,
            truncated=truncated,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return the current internal state."""
        return self._state.model_copy(deep=True)

    def grade(self) -> dict:
        """Run the grader on the current state."""
        if self._task is None:
            return {"error": "No task loaded."}
        return grade_task(self._state, self._task)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> None:
        """Mutate internal state based on action."""
        at = action.action_type
        eid = action.email_id
        payload = action.payload

        if at == "open_email":
            if eid:
                for e in self._state.emails:
                    if e.email_id == eid:
                        e.is_read = True
                        self._state.current_email_id = eid
                        break

        elif at == "classify_email":
            if eid:
                self._state.classifications[eid] = payload.get("category", "unknown")

        elif at == "set_priority":
            if eid:
                self._state.priorities[eid] = payload.get("priority", "medium")

        elif at == "draft_reply":
            if eid:
                self._state.drafts[eid] = payload.get("reply_text", "")

        elif at == "send_reply":
            if eid:
                # Use draft if available, otherwise use payload
                text = self._state.drafts.get(eid, payload.get("reply_text", ""))
                self._state.sent_replies[eid] = text

        elif at == "escalate":
            if eid:
                self._state.escalations[eid] = payload.get("team", "general")

        elif at == "mark_spam":
            if eid and eid not in self._state.spam_flags:
                self._state.spam_flags.append(eid)

        elif at == "archive":
            if eid and eid not in self._state.archives:
                self._state.archives.append(eid)

        elif at == "schedule_followup":
            if eid and eid not in self._state.followups:
                self._state.followups.append(eid)

        elif at == "request_more_info":
            if eid and eid not in self._state.info_requests:
                self._state.info_requests.append(eid)

        elif at == "finish":
            pass  # Handled in step()

    def _build_observation(self) -> Observation:
        """Construct the observation from current state."""
        inbox_summary = []
        for e in self._state.emails:
            inbox_summary.append(
                EmailSummary(
                    email_id=e.email_id,
                    sender=e.sender,
                    subject=e.subject,
                    is_read=e.is_read,
                    priority=self._state.priorities.get(e.email_id),
                    classification=self._state.classifications.get(e.email_id),
                )
            )

        current_email = None
        if self._state.current_email_id:
            for e in self._state.emails:
                if e.email_id == self._state.current_email_id:
                    current_email = e
                    break

        # Pending items: emails that haven't been fully processed
        pending = []
        for e in self._state.emails:
            eid = e.email_id
            has_class = eid in self._state.classifications
            has_prio = eid in self._state.priorities
            has_action = (
                eid in self._state.sent_replies
                or eid in self._state.escalations
                or eid in self._state.spam_flags
                or eid in self._state.archives
            )
            if not (has_class and has_prio and has_action):
                pending.append(eid)

        available_actions = [
            "open_email", "classify_email", "set_priority",
            "draft_reply", "send_reply", "escalate",
            "mark_spam", "archive", "schedule_followup",
            "request_more_info", "finish",
        ]

        return Observation(
            task_id=self._state.task_id,
            goal=self._task.goal if self._task else "",
            instruction=self._task.instruction if self._task else "",
            inbox_summary=inbox_summary,
            current_email=current_email,
            available_actions=available_actions,
            action_history=[
                {"step": h["step"], "action_type": h["action_type"], "email_id": h.get("email_id")}
                for h in self._state.action_history[-10:]  # last 10 for context
            ],
            pending_items=pending,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
        )

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict."""
        info: dict[str, Any] = {
            "step_count": self._state.step_count,
            "total_reward": round(self._state.total_reward, 4),
            "emails_classified": len(self._state.classifications),
            "emails_prioritized": len(self._state.priorities),
            "replies_sent": len(self._state.sent_replies),
            "escalations": len(self._state.escalations),
            "spam_flagged": len(self._state.spam_flags),
            "archived": len(self._state.archives),
        }
        if self._state.done:
            info["success"] = self._state.success
        return info
