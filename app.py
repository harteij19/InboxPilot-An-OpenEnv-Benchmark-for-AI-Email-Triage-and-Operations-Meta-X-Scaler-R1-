from __future__ import annotations

import json

import gradio as gr

from inboxpilot.agent import TrainedAgent, UntrainedAgent
from inboxpilot.environment import InboxPilotEnvironment
from inboxpilot.reward import compute_total_reward
from inboxpilot.sample_data import get_sample_emails


def _pretty(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _run_agent(mode: str) -> tuple[str, str, str]:
    emails = get_sample_emails()
    env = InboxPilotEnvironment(emails=emails)
    env.reset()

    if mode == "untrained":
        agent = UntrainedAgent()
    else:
        agent = TrainedAgent()

    actions = agent.run(emails)
    for action in actions:
        env.step({"email_id": action["email_id"], "action": action["action"]})

    total_reward, breakdown = compute_total_reward(actions)

    summary = {
        "agent": agent.name,
        "total_reward": total_reward,
        "processed_count": len(actions),
    }
    return _pretty(actions), _pretty(breakdown), _pretty(summary)


def run_untrained() -> tuple[str, str, str]:
    return _run_agent("untrained")


def run_trained() -> tuple[str, str, str]:
    return _run_agent("trained")


def compare_rewards() -> str:
    emails = get_sample_emails()
    untrained_actions = UntrainedAgent().run(emails)
    trained_actions = TrainedAgent().run(emails)

    untrained_reward, _ = compute_total_reward(untrained_actions)
    trained_reward, _ = compute_total_reward(trained_actions)

    delta = round(trained_reward - untrained_reward, 2)
    lines = [
        f"Untrained reward: {untrained_reward}",
        f"Trained reward: {trained_reward}",
        f"Improvement: {delta}",
    ]
    return "\n".join(lines)


with gr.Blocks(title="InboxPilot Demo") as demo:
    gr.Markdown("# InboxPilot Demo (CPU-Only)")
    gr.Markdown("Lightweight OpenEnv-style email triage demo with no paid APIs and no model downloads.")

    sample_box = gr.Textbox(label="Sample Emails", lines=16, value=_pretty(get_sample_emails()))

    with gr.Row():
        run_untrained_btn = gr.Button("Run Untrained Agent", variant="secondary")
        run_trained_btn = gr.Button("Run Trained Agent", variant="primary")

    actions_box = gr.Textbox(label="Agent Decisions", lines=16)
    reward_box = gr.Textbox(label="Reward Breakdown", lines=10)
    summary_box = gr.Textbox(label="Run Summary", lines=6)
    compare_box = gr.Textbox(label="Before vs After Reward", lines=4, value=compare_rewards())

    run_untrained_btn.click(
        fn=run_untrained,
        outputs=[actions_box, reward_box, summary_box],
    )

    run_trained_btn.click(
        fn=run_trained,
        outputs=[actions_box, reward_box, summary_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
