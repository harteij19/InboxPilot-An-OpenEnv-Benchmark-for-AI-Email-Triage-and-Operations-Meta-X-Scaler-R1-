"""Before-vs-after demo for InboxPilot decision agents."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.decision_agent import BaselineAgent, TrainedAgent
from environment.email_env import EmailEnv


def _print_report(title: str, report: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Agent: {report['agent']}")

    print("\nPriority scores:")
    for email_id, meta in report["priorities"].items():
        print(f"- {email_id}: score={meta['score']} | {meta['reason']}")

    print("\nPipeline trace (email -> classify -> prioritize -> action):")
    for row in report["decision_trace"]:
        print(f"- {row}")

    print("\nExplainability action logs:")
    for row in report["action_logs"]:
        print(row)

    print("\nCalendar state:")
    print(f"- scheduled: {report['calendar']['scheduled']}")
    print(f"- rejected: {report['calendar']['rejected']}")

    print("\nReward summary:")
    print(f"- total: {report['reward_total']}")
    print(f"- breakdown: {report['reward_breakdown']}")


def main() -> None:
    now = datetime.now(timezone.utc)
    inbox = EmailEnv.demo_inbox(now=now)

    baseline_env = EmailEnv(inbox=inbox, now=now)
    trained_env = EmailEnv(inbox=inbox, now=now)

    baseline_report = baseline_env.run_episode(BaselineAgent())
    trained_report = trained_env.run_episode(TrainedAgent())

    print("InboxPilot Demo: baseline_agent vs trained_agent")
    print("Scenario: Boss report, client meeting, friend party, spam iPhone email")

    _print_report("BEFORE", baseline_report)
    _print_report("AFTER", trained_report)

    delta = round(trained_report["reward_total"] - baseline_report["reward_total"], 2)
    print("\n=== Comparison ===")
    print(f"Baseline total reward: {baseline_report['reward_total']}")
    print(f"Trained total reward:  {trained_report['reward_total']}")
    print(f"Reward improvement:    {delta}")


if __name__ == "__main__":
    main()
