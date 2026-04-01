"""
Tests for the InboxPilot deterministic graders.
"""

import pytest
from app.env import InboxPilotEnv
from app.graders import grade_task
from app.models import Action
from app.tasks import get_all_task_ids, get_task


class TestGraderScoreRange:
    """Test that all graders return scores in [0.0, 1.0]."""

    @pytest.mark.parametrize("task_id", get_all_task_ids())
    def test_grader_score_range_empty(self, task_id):
        """Grading an empty episode should return a valid score."""
        env = InboxPilotEnv()
        env.reset(task_id)
        result = grade_task(env.state(), get_task(task_id))
        assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.parametrize("task_id", get_all_task_ids())
    def test_grader_score_range_after_actions(self, task_id):
        """Grading after some actions should return a valid score."""
        env = InboxPilotEnv()
        env.reset(task_id)
        emails = env.state().emails
        if emails:
            env.step(Action(action_type="open_email", email_id=emails[0].email_id))
            env.step(Action(action_type="classify_email", email_id=emails[0].email_id, payload={"category": "test"}))
        result = grade_task(env.state(), get_task(task_id))
        assert 0.0 <= result["score"] <= 1.0

    def test_perfect_easy_task(self):
        """A perfect run of the easy task should score > 0.8."""
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        task = get_task("easy_support_triage")

        for ak in task.answer_keys:
            env.step(Action(action_type="open_email", email_id=ak.email_id))
            env.step(Action(action_type="classify_email", email_id=ak.email_id, payload={"category": ak.classification}))
            env.step(Action(action_type="set_priority", email_id=ak.email_id, payload={"priority": ak.priority}))

            if ak.action == "escalate":
                env.step(Action(action_type="escalate", email_id=ak.email_id, payload={"team": ak.escalation_target}))
            elif ak.action == "mark_spam":
                env.step(Action(action_type="mark_spam", email_id=ak.email_id))
            elif ak.action == "archive":
                env.step(Action(action_type="archive", email_id=ak.email_id))
            elif ak.action == "send_reply":
                reply_text = " ".join(ak.reply_required_keywords) if ak.reply_required_keywords else "Acknowledged."
                env.step(Action(action_type="draft_reply", email_id=ak.email_id, payload={"reply_text": reply_text}))
                env.step(Action(action_type="send_reply", email_id=ak.email_id))

            if ak.followup_required:
                env.step(Action(action_type="schedule_followup", email_id=ak.email_id))

        result = grade_task(env.state(), task)
        assert result["score"] > 0.8, f"Perfect run scored {result['score']}, expected > 0.8"

    def test_empty_state_scores_low(self):
        """An empty state with no actions should score low."""
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        result = grade_task(env.state(), get_task("easy_support_triage"))
        assert result["score"] < 0.3


class TestGraderDeterminism:
    """Test that graders are deterministic."""

    @pytest.mark.parametrize("task_id", get_all_task_ids())
    def test_same_state_same_score(self, task_id):
        """Grading the same state twice should produce the same score."""
        env = InboxPilotEnv()
        env.reset(task_id)
        task = get_task(task_id)
        r1 = grade_task(env.state(), task)
        r2 = grade_task(env.state(), task)
        assert r1["score"] == r2["score"]


class TestGraderComponents:
    """Test individual grading components."""

    def test_classification_scoring(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="classify_email", email_id="email_006", payload={"category": "spam"}))
        result = grade_task(env.state(), get_task("easy_support_triage"))
        # Check that per-email classification is scored
        assert "email_006" in result["per_email"]
        assert result["per_email"]["email_006"]["components"]["classification"] == 1.0

    def test_wrong_classification_scores_zero(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="classify_email", email_id="email_006", payload={"category": "billing"}))
        result = grade_task(env.state(), get_task("easy_support_triage"))
        assert result["per_email"]["email_006"]["components"]["classification"] == 0.0
