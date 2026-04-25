from __future__ import annotations

import json
import re
from typing import Any

import gradio as gr


DEFAULT_EMAILS: list[dict[str, str]] = [
    {"sender": "Bank", "text": "Your OTP is 839201. Do not share this verification code."},
    {"sender": "Boss", "text": "Need report by EOD. Client meeting tomorrow morning."},
    {"sender": "Friend", "text": "Can you help me? My family is in trouble and I need urgent help."},
    {"sender": "Friend", "text": "Party tonight? Weekend hangout plans?"},
    {"sender": "HR", "text": "Please submit onboarding documents and policy form."},
    {"sender": "Promo", "text": "You win free lottery rewards. Click here now."},
]

CRITICAL_PATTERNS = [
    "otp",
    "verification code",
    "transaction",
    "account alert",
    "password",
    "login",
    "bank",
]
URGENT_WORK_PATTERNS = [
    "deadline",
    "eod",
    "urgent",
    "asap",
    "client meeting",
    "meeting",
    "client",
    "report",
]
EMOTIONAL_PATTERNS = [
    "need help",
    "emergency",
    "family problem",
    "loan",
    "medical",
    "urgent help",
    "family is in trouble",
]
TASK_PATTERNS = [
    "submit",
    "update",
    "review",
    "document",
    "form",
    "policy",
]
CASUAL_PATTERNS = [
    "party",
    "hangout",
    "weekend",
    "hello",
    "hey",
]
SPAM_PATTERNS = [
    "win",
    "lottery",
    "free",
    "click",
    "click here",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def load_data(file_obj: Any) -> tuple[list[dict[str, str]], str]:
    if file_obj is None:
        return DEFAULT_EMAILS, "Using default dataset"

    try:
        if isinstance(file_obj, bytes):
            payload = json.loads(file_obj.decode("utf-8"))
        elif isinstance(file_obj, str):
            try:
                with open(file_obj, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except OSError:
                payload = json.loads(file_obj)
        elif isinstance(file_obj, dict) and isinstance(file_obj.get("path"), str):
            with open(file_obj["path"], "r", encoding="utf-8") as f:
                payload = json.load(f)
        elif hasattr(file_obj, "read"):
            raw = file_obj.read()
            if isinstance(raw, bytes):
                payload = json.loads(raw.decode("utf-8"))
            else:
                payload = json.loads(str(raw))
        else:
            return DEFAULT_EMAILS, "Unsupported upload format, fallback to default dataset"
    except Exception:
        return DEFAULT_EMAILS, "Invalid JSON, fallback to default dataset"

    if not isinstance(payload, list) or len(payload) == 0:
        return DEFAULT_EMAILS, "Empty/invalid JSON list, fallback to default dataset"

    cleaned: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            return DEFAULT_EMAILS, "Invalid item format, fallback to default dataset"
        sender = str(item.get("sender", "")).strip()
        text = str(item.get("text", "")).strip()
        if not sender or not text:
            return DEFAULT_EMAILS, "Missing sender/text values, fallback to default dataset"
        cleaned.append({"sender": sender, "text": text})

    return cleaned, "Using uploaded dataset"


def detect_intent(text: str) -> tuple[str, dict[str, bool], list[str]]:
    t = _normalize(text)
    flags = {
        "critical": _contains_any(t, CRITICAL_PATTERNS),
        "urgent_work": _contains_any(t, URGENT_WORK_PATTERNS),
        "emotional": _contains_any(t, EMOTIONAL_PATTERNS),
        "task": _contains_any(t, TASK_PATTERNS),
        "casual": _contains_any(t, CASUAL_PATTERNS),
        "spam": _contains_any(t, SPAM_PATTERNS),
    }

    reasons: list[str] = []
    if flags["critical"]:
        reasons.append("Detected OTP/banking/security content")
    if flags["urgent_work"]:
        reasons.append("Detected urgent work/deadline context")
    if flags["emotional"]:
        reasons.append("Detected emotional distress/help request")
    if flags["task"]:
        reasons.append("Detected task/admin request")
    if flags["casual"]:
        reasons.append("Detected casual conversation")
    if flags["spam"]:
        reasons.append("Detected spam/scam language")

    if flags["critical"]:
        primary = "critical"
    elif flags["urgent_work"]:
        primary = "urgent_work"
    elif flags["emotional"]:
        primary = "emotional"
    elif flags["task"]:
        primary = "task"
    elif flags["spam"]:
        primary = "spam"
    elif flags["casual"]:
        primary = "casual"
    else:
        primary = "task"

    return primary, flags, reasons


def score_email(text: str) -> tuple[int, str, dict[str, bool], list[str]]:
    primary_intent, flags, reasons = detect_intent(text)

    if flags["critical"]:
        return 10, primary_intent, flags, reasons

    score = 0
    if flags["urgent_work"]:
        score += 5
    if flags["emotional"]:
        score += 4
    if flags["task"]:
        score += 2
    if flags["casual"]:
        score -= 2
    if flags["spam"]:
        score -= 3

    return score, primary_intent, flags, reasons


def map_priority(score: int, sender: str, flags: dict[str, bool]) -> str:
    s = _normalize(sender)

    if flags["critical"]:
        return "HIGH"
    if "hr" in s and not flags["urgent_work"] and not flags["critical"] and not flags["spam"]:
        return "MEDIUM"
    if "friend" in s and flags["casual"] and not flags["emotional"]:
        return "LOW"
    if flags["emotional"] and score < 4:
        return "MEDIUM"

    if score >= 8:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def _reply(priority: str, sender: str) -> str:
    if priority == "HIGH":
        return f"Hi {sender}, I understand the urgency of this message and will take immediate action."
    if priority == "MEDIUM":
        return f"Hi {sender}, I’ve noted this and will address it shortly."
    return f"Hi {sender}, thanks for the update. I will review it when possible."


def run_untrained(file_obj: Any) -> str:
    emails, source_note = load_data(file_obj)
    lines = ["## ❌ Untrained Output", f"**Source:** {source_note}", "---"]

    for idx, e in enumerate(emails, start=1):
        sender = e["sender"]
        text = _normalize(e["text"])

        if any(k in text for k in ["otp", "bank", "transaction", "urgent", "deadline", "eod"]):
            pred = "LOW"
        elif any(k in text for k in ["party", "weekend", "hangout"]):
            pred = "HIGH"
        else:
            pred = ["LOW", "MEDIUM", "HIGH"][(idx + len(text)) % 3]

        reply = "ok noted maybe later"

        lines.append(
            f"📩 **{sender}**\n"
            f"Priority: **{pred}**\n"
            f"Score: **N/A**\n\n"
            f"Reply: {reply}\n\n"
            f"Message: {e['text'][:180]}{'...' if len(e['text']) > 180 else ''}\n"
            "---"
        )

    return "\n".join(lines)


def run_trained(file_obj: Any) -> str:
    emails, source_note = load_data(file_obj)
    lines = ["## ✅ Trained Output", f"**Source:** {source_note}", "---"]

    high_count = 0
    med_count = 0
    low_count = 0

    for e in emails:
        sender = e["sender"]
        text = e["text"]

        score, intent, flags, reasons = score_email(text)
        priority = map_priority(score, sender, flags)

        if priority == "HIGH":
            high_count += 1
        elif priority == "MEDIUM":
            med_count += 1
        else:
            low_count += 1

        reason_text = "; ".join(reasons) if reasons else "No strong signal detected"

        lines.append(
            f"📩 **{sender}**\n"
            f"Priority: **{priority}**\n"
            f"Score: **{score}**\n"
            f"\nReply: {_reply(priority, sender)}\n"
            f"Reason: {reason_text}\n"
            "---"
        )

    lines.extend(
        [
            "## 📊 Distribution Summary",
            f"- Total emails processed: **{len(emails)}**",
            f"- HIGH: **{high_count}**",
            f"- MEDIUM: **{med_count}**",
            f"- LOW: **{low_count}**",
        ]
    )

    return "\n".join(lines)


def _inbox_preview(emails: list[dict[str, str]]) -> str:
    parts = ["## 📬 Inbox Preview", ""]
    for e in emails:
        preview = e["text"][:160] + ("..." if len(e["text"]) > 160 else "")
        parts.append(f"📩 **{e['sender']}**\n**Message:** {preview}\n\n---\n")
    return "\n".join(parts)


with gr.Blocks(title="📧 InboxPilot – AI Email Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 InboxPilot – AI Email Assistant")
    gr.Markdown("*Context-aware email prioritization demo*")
    gr.Markdown("---")

    gr.Markdown(_inbox_preview(DEFAULT_EMAILS))
    gr.Markdown("---")

    gr.Markdown("## ⚙️ Run Agent")
    gr.Markdown("Upload a JSON list like: [{'sender':'...','text':'...'}]")
    data_input = gr.File(label="Email Dataset JSON", file_types=[".json"], type="binary")

    with gr.Row():
        btn_untrained = gr.Button("❌ Run Untrained Agent", variant="secondary")
        btn_trained = gr.Button("✅ Run Trained Agent", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## 📊 Results")
    with gr.Row():
        with gr.Column(scale=1):
            untrained_output = gr.Markdown("Untrained output will appear here.")
        with gr.Column(scale=1):
            trained_output = gr.Markdown("Trained output will appear here.")

    btn_untrained.click(
        fn=run_untrained,
        inputs=[data_input],
        outputs=[untrained_output],
    )

    btn_trained.click(
        fn=run_trained,
        inputs=[data_input],
        outputs=[trained_output],
    )


if __name__ == "__main__":
    demo.launch()
