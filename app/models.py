"""
InboxPilot — Pydantic models for the OpenEnv environment.

Defines typed Observation, Action, Reward, and EnvironmentState models
used across the environment, server, graders, and inference script.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    OPEN_EMAIL = "open_email"
    CLASSIFY_EMAIL = "classify_email"
    SET_PRIORITY = "set_priority"
    DRAFT_REPLY = "draft_reply"
    SEND_REPLY = "send_reply"
    ESCALATE = "escalate"
    MARK_SPAM = "mark_spam"
    ARCHIVE = "archive"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    REQUEST_MORE_INFO = "request_more_info"
    FINISH = "finish"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """Represents a single email in the inbox."""
    email_id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    is_read: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """An action the agent can take."""
    action_type: str
    email_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Structured reward returned after each step."""
    score: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailSummary(BaseModel):
    """Short summary of an email for the inbox view."""
    email_id: str
    sender: str
    subject: str
    is_read: bool
    priority: Optional[str] = None
    classification: Optional[str] = None


class Observation(BaseModel):
    """Observation returned to the agent after reset / step."""
    task_id: str
    goal: str
    instruction: str
    inbox_summary: list[EmailSummary] = Field(default_factory=list)
    current_email: Optional[Email] = None
    available_actions: list[str] = Field(default_factory=list)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    pending_items: list[str] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 50


# ---------------------------------------------------------------------------
# Environment State (internal)
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full internal state of the environment."""
    task_id: str = ""
    emails: list[Email] = Field(default_factory=list)
    current_email_id: Optional[str] = None
    classifications: dict[str, str] = Field(default_factory=dict)
    priorities: dict[str, str] = Field(default_factory=dict)
    drafts: dict[str, str] = Field(default_factory=dict)
    sent_replies: dict[str, str] = Field(default_factory=dict)
    escalations: dict[str, str] = Field(default_factory=dict)
    spam_flags: list[str] = Field(default_factory=list)
    archives: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    info_requests: list[str] = Field(default_factory=list)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    success: bool = False
    step_count: int = 0
    max_steps: int = 50
    total_reward: float = 0.0


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Return value of env.step() — matches Gymnasium-style 5-tuple."""
    observation: Observation
    reward: Reward
    done: bool
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task definition (loaded from JSON)
# ---------------------------------------------------------------------------

class AnswerKey(BaseModel):
    """Expected correct actions per email."""
    email_id: str
    classification: str
    priority: str
    action: str  # expected action: escalate / archive / mark_spam / send_reply
    escalation_target: Optional[str] = None
    reply_required_keywords: list[str] = Field(default_factory=list)
    followup_required: bool = False


class TaskDefinition(BaseModel):
    """Complete task loaded from JSON."""
    task_id: str
    name: str
    description: str
    goal: str
    instruction: str
    difficulty: str
    max_steps: int = 50
    emails: list[Email]
    answer_keys: list[AnswerKey]
