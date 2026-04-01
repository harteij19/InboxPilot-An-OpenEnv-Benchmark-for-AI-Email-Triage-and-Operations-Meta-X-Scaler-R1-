"""
Tests for the InboxPilot core environment.
"""

import pytest
from app.env import InboxPilotEnv
from app.models import Action
from app.tasks import get_all_task_ids


class TestEnvironmentReset:
    """Test that reset() produces a valid initial state."""

    def test_reset_returns_step_result(self):
        env = InboxPilotEnv()
        result = env.reset("easy_support_triage")
        assert result.done is False
        assert result.observation.task_id == "easy_support_triage"
        assert result.observation.step_count == 0
        assert len(result.observation.inbox_summary) == 6

    def test_reset_clears_state(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_001"))
        # Reset should clear everything
        result = env.reset("easy_support_triage")
        assert result.observation.step_count == 0
        state = env.state()
        assert len(state.action_history) == 0
        assert len(state.classifications) == 0

    @pytest.mark.parametrize("task_id", get_all_task_ids())
    def test_reset_all_tasks(self, task_id):
        env = InboxPilotEnv()
        result = env.reset(task_id)
        assert result.done is False
        assert result.observation.task_id == task_id
        assert len(result.observation.inbox_summary) > 0

    def test_reset_invalid_task(self):
        env = InboxPilotEnv()
        with pytest.raises(KeyError):
            env.reset("nonexistent_task")


class TestEnvironmentStep:
    """Test that step() correctly updates state."""

    def test_open_email_updates_state(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        result = env.step(Action(action_type="open_email", email_id="email_001"))
        assert result.done is False
        assert result.observation.step_count == 1
        state = env.state()
        assert state.current_email_id == "email_001"

    def test_classify_email(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_001"))
        env.step(Action(action_type="classify_email", email_id="email_001", payload={"category": "refund"}))
        state = env.state()
        assert state.classifications.get("email_001") == "refund"

    def test_set_priority(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_001"))
        env.step(Action(action_type="set_priority", email_id="email_001", payload={"priority": "high"}))
        state = env.state()
        assert state.priorities.get("email_001") == "high"

    def test_escalate(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_001"))
        env.step(Action(action_type="escalate", email_id="email_001", payload={"team": "billing"}))
        state = env.state()
        assert state.escalations.get("email_001") == "billing"

    def test_mark_spam(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="mark_spam", email_id="email_006"))
        state = env.state()
        assert "email_006" in state.spam_flags

    def test_draft_and_send_reply(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_003"))
        env.step(Action(action_type="draft_reply", email_id="email_003", payload={"reply_text": "Sure, let's reschedule to Thursday."}))
        env.step(Action(action_type="send_reply", email_id="email_003"))
        state = env.state()
        assert "email_003" in state.sent_replies
        assert "thursday" in state.sent_replies["email_003"].lower()

    def test_finish_ends_episode(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        result = env.step(Action(action_type="finish"))
        assert result.done is True
        state = env.state()
        assert state.done is True

    def test_step_after_done(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="finish"))
        result = env.step(Action(action_type="open_email", email_id="email_001"))
        assert result.done is True  # should stay done

    def test_step_without_reset_raises(self):
        env = InboxPilotEnv()
        with pytest.raises(RuntimeError):
            env.step(Action(action_type="open_email", email_id="email_001"))

    def test_max_steps_terminates(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        # Force max_steps to something small
        env._state.max_steps = 3
        for _ in range(3):
            result = env.step(Action(action_type="open_email", email_id="email_001"))
        assert result.done is True


class TestEnvironmentState:
    """Test that state() returns correct data."""

    def test_state_returns_copy(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        s1 = env.state()
        s2 = env.state()
        assert s1 is not s2  # should be a copy

    def test_state_has_required_fields(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        state = env.state()
        assert hasattr(state, "task_id")
        assert hasattr(state, "emails")
        assert hasattr(state, "classifications")
        assert hasattr(state, "priorities")
        assert hasattr(state, "done")
        assert hasattr(state, "step_count")


class TestRewardSignals:
    """Test that rewards are shaped and meaningful."""

    def test_correct_classification_positive_reward(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_006"))
        result = env.step(Action(action_type="classify_email", email_id="email_006", payload={"category": "spam"}))
        assert result.reward.score > 0

    def test_wrong_classification_negative_reward(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        env.step(Action(action_type="open_email", email_id="email_006"))
        result = env.step(Action(action_type="classify_email", email_id="email_006", payload={"category": "billing"}))
        assert result.reward.score < 0

    def test_invalid_email_negative_reward(self):
        env = InboxPilotEnv()
        env.reset("easy_support_triage")
        result = env.step(Action(action_type="open_email", email_id="nonexistent"))
        assert result.reward.score < 0
