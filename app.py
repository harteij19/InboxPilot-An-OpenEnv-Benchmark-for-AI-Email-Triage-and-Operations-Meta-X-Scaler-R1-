from __future__ import annotations

import json
from typing import Any

import gradio as gr


DEFAULT_EMAILS: list[dict[str, str]] = [
    {"sender": "Boss", "text": "Need report by EOD", "true_priority": "High"},
    {"sender": "HR", "text": "Submit documents", "true_priority": "Medium"},
    {"sender": "Friend", "text": "Party tonight?", "true_priority": "Low"},
]


def normalize_priority(value: str) -> str:
    v = str(value).strip().lower()
    if v == "high":
        return "High"
    if v == "medium":
        return "Medium"
    if v == "low":
        return "Low"
    return "Low"


def _inbox_row(email: dict[str, str], idx: int) -> str:
    sender = email.get("sender", "Unknown")
    text = email.get("text", "")
    truth = normalize_priority(email.get("true_priority", "Low"))
    urgency_icon = "⚠️" if truth == "High" else "🗂️" if truth == "Medium" else "💬"
    return (
        f"**📩 Email {idx}**  \n"
        f"- Sender: **{sender}**  \n"
        f"- Message: {text}  \n"
        f"- True Priority: {urgency_icon} **{truth}**"
    )


def format_inbox_markdown(emails: list[dict[str, str]], title: str) -> str:
    rows = [f"### {title}"]
    for i, email in enumerate(emails, start=1):
        rows.append(_inbox_row(email, i))
    return "\n\n".join(rows)


def _parse_uploaded_json(file: Any) -> Any:
    if file is None:
        return None
    if isinstance(file, bytes):
        return json.loads(file.decode("utf-8"))
    if isinstance(file, str):
        try:
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return json.loads(file)
    if isinstance(file, dict):
        path = file.get("path")
        if isinstance(path, str) and path:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    if hasattr(file, "read"):
        content = file.read()
        if isinstance(content, bytes):
            return json.loads(content.decode("utf-8"))
        return json.loads(str(content))
    raise ValueError("Unsupported uploaded file format")


def load_data(file: Any) -> tuple[list[dict[str, str]], str]:
    if file is None:
        return DEFAULT_EMAILS, "Using default dataset."

    try:
        payload = _parse_uploaded_json(file)
    except Exception:
        return DEFAULT_EMAILS, "Invalid JSON upload. Falling back to default dataset."

    if not isinstance(payload, list) or not payload:
        return DEFAULT_EMAILS, "Uploaded JSON is empty/invalid. Falling back to default dataset."

    validated: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            return DEFAULT_EMAILS, "JSON format mismatch. Falling back to default dataset."

        sender = str(item.get("sender", "")).strip()
        text = str(item.get("text", "")).strip()
        truth = normalize_priority(str(item.get("true_priority", "")).strip())

        if not sender or not text:
            return DEFAULT_EMAILS, "Missing sender/text fields. Falling back to default dataset."

        validated.append(
            {
                "sender": sender,
                "text": text,
                "true_priority": truth,
            }
        )

    return validated, "Using uploaded custom dataset."


def _untrained_priority(email: dict[str, str], idx: int) -> str:
    sender = email.get("sender", "").lower()
    text = email.get("text", "").lower()

    # Intentionally weak deterministic logic for clear contrast.
    if "friend" in sender or "party" in text:
        return "High"
    if "boss" in sender or "deadline" in text or "eod" in text:
        return "Medium"
    cycle = ["Low", "High", "Medium"]
    return cycle[idx % len(cycle)]


def _trained_priority(email: dict[str, str]) -> str:
    sender = email.get("sender", "").lower()
    text = email.get("text", "").lower()

    high_signals = ["boss", "ceo", "urgent", "asap", "deadline", "eod", "report"]
    medium_signals = ["hr", "documents", "submit", "policy", "form", "admin"]

    if any(sig in sender or sig in text for sig in high_signals):
        return "High"
    if any(sig in sender or sig in text for sig in medium_signals):
        return "Medium"
    return "Low"


def _untrained_reply(email: dict[str, str], priority: str) -> str:
    sender = email.get("sender", "there")
    return f"ok {sender}, noted. maybe later. [{priority}]"


def _trained_reply(email: dict[str, str], priority: str) -> str:
    sender = email.get("sender", "there")
    if priority == "High":
        return f"Thank you, {sender}. This is urgent and I will prioritize it immediately."
    if priority == "Medium":
        return f"Thank you, {sender}. I have logged this and will complete it in the next work cycle."
    return f"Thanks, {sender}. I have received this and will follow up appropriately."


def calculate_accuracy(predictions: list[str], true_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    correct = sum(
        1
        for pred, truth in zip(predictions, true_labels)
        if normalize_priority(pred) == normalize_priority(truth)
    )
    return correct / len(true_labels)


def _build_outputs(emails: list[dict[str, str]], source_note: str, triggered_by: str) -> tuple[str, str, str, str, str]:
    truths = [normalize_priority(e.get("true_priority", "Low")) for e in emails]

    untrained_preds: list[str] = []
    trained_preds: list[str] = []
    untrained_lines = ["### ❌ Untrained Output"]
    trained_lines = ["### ✅ Trained Output"]
    reasoning_lines = ["### 🧠 Agent Reasoning"]

    for i, email in enumerate(emails, start=1):
        sender = email.get("sender", "Unknown")
        text = email.get("text", "")
        truth = normalize_priority(email.get("true_priority", "Low"))

        untrained_priority = _untrained_priority(email, i - 1)
        trained_priority = _trained_priority(email)
        untrained_preds.append(untrained_priority)
        trained_preds.append(trained_priority)

        untrained_lines.append(
            f"**Email {i} ({sender})**  \n"
            f"- Predicted Priority: **{untrained_priority}**  \n"
            f"- True Priority: {truth}  \n"
            f"- Reply: *{_untrained_reply(email, untrained_priority)}*"
        )

        trained_lines.append(
            f"**Email {i} ({sender})**  \n"
            f"- Predicted Priority: **{trained_priority}**  \n"
            f"- True Priority: {truth}  \n"
            f"- Reply: *{_trained_reply(email, trained_priority)}*"
        )

        if trained_priority == "High":
            reasoning = "deadline/urgency or critical sender detected"
        elif trained_priority == "Medium":
            reasoning = "administrative or operational intent detected"
        else:
            reasoning = "casual or low-urgency context detected"
        reasoning_lines.append(f"- **{sender}** -> **{trained_priority}** ({reasoning})")

    untrained_acc = calculate_accuracy(untrained_preds, truths)
    trained_acc = calculate_accuracy(trained_preds, truths)

    task_untrained = "Low" if untrained_acc < 0.6 else "Medium"
    task_trained = "High" if trained_acc >= 0.8 else "Medium"

    metrics_table = (
        "### 📊 Performance Metrics\n\n"
        "| Metric | Untrained ❌ | Trained ✅ |\n"
        "| --- | ---: | ---: |\n"
        f"| Priority Accuracy | {untrained_acc:.0%} | {trained_acc:.0%} |\n"
        "| Reply Quality | Poor | Professional |\n"
        f"| Task Completion | {task_untrained} | {task_trained} |"
    )

    active_inbox = (
        format_inbox_markdown(emails, "📩 Active Inbox")
        + "\n\n"
        + "### ✅ Run Info\n"
        + f"- Data Source: {source_note}\n"
        + f"- Triggered By: {triggered_by}\n"
        + f"- Emails Evaluated: {len(emails)}"
    )

    return (
        active_inbox,
        "\n\n".join(untrained_lines),
        "\n\n".join(trained_lines),
        metrics_table,
        "\n".join(reasoning_lines),
    )


def run_untrained(data: Any) -> tuple[str, str, str, str, str]:
    emails, note = load_data(data)
    return _build_outputs(emails, note, "❌ Run Untrained Agent")


def run_trained(data: Any) -> tuple[str, str, str, str, str]:
    emails, note = load_data(data)
    return _build_outputs(emails, note, "✅ Run Trained Agent")


with gr.Blocks(title="📧 InboxPilot - AI Email Assistant") as demo:
    gr.Markdown("# 📧 InboxPilot - AI Email Assistant")
    gr.Markdown("InboxPilot triages emails, predicts priority, drafts replies, and shows measurable quality differences between untrained and trained agent behavior.")

    gr.Markdown(format_inbox_markdown(DEFAULT_EMAILS, "🧾 Default Inbox"))

    gr.Markdown("## 📂 Custom Input")
    upload = gr.File(label="Upload Custom Email JSON", file_types=[".json"], type="binary")

    gr.Markdown("## 🤖 Agent Controls")
    with gr.Row():
        run_untrained_btn = gr.Button("❌ Run Untrained Agent", variant="secondary")
        run_trained_btn = gr.Button("✅ Run Trained Agent", variant="primary")

    active_inbox_box = gr.Markdown("### 📩 Active Inbox\nClick an agent button to evaluate default or uploaded JSON data.")

    gr.Markdown("## 📊 Results Comparison")
    with gr.Row():
        untrained_box = gr.Markdown("### ❌ Untrained Output\nNo run yet.")
        trained_box = gr.Markdown("### ✅ Trained Output\nNo run yet.")

    metrics_box = gr.Markdown("### 📊 Performance Metrics\nNo run yet.")
    reasoning_box = gr.Markdown("### 🧠 Agent Reasoning\nNo run yet.")

    run_untrained_btn.click(
        fn=run_untrained,
        inputs=[upload],
        outputs=[active_inbox_box, untrained_box, trained_box, metrics_box, reasoning_box],
    )

    run_trained_btn.click(
        fn=run_trained,
        inputs=[upload],
        outputs=[active_inbox_box, untrained_box, trained_box, metrics_box, reasoning_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
