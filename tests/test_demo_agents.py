"""Tests for the before/after decision-agent demo workflow."""

from __future__ import annotations

from datetime import datetime, timezone

from agents.decision_agent import BaselineAgent, TrainedAgent
from environment.email_env import EmailEnv


def test_trained_agent_handles_priority_and_spam() -> None:
    now = datetime.now(timezone.utc)
    env = EmailEnv(inbox=EmailEnv.demo_inbox(now), now=now)

    report = env.run_episode(TrainedAgent())

    assert report["processed_order"][0] == "email_001"

    action_map = {entry["email_id"]: entry["action"] for entry in report["action_logs"]}
    assert action_map["email_004"] == "ignore"
    assert action_map["email_002"] == "schedule"


def test_trained_agent_outperforms_baseline() -> None:
    now = datetime.now(timezone.utc)
    inbox = EmailEnv.demo_inbox(now)

    baseline_report = EmailEnv(inbox=inbox, now=now).run_episode(BaselineAgent())
    trained_report = EmailEnv(inbox=inbox, now=now).run_episode(TrainedAgent())

    assert trained_report["reward_total"] > baseline_report["reward_total"]
