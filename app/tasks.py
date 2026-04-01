"""
InboxPilot — Task loader.

Loads task definitions from JSON files in the data/tasks/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import TaskDefinition

logger = logging.getLogger("inboxpilot")

# Path to data/tasks relative to the project root
_TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks"

# Cache loaded tasks
_task_cache: dict[str, TaskDefinition] = {}


def _load_tasks() -> None:
    """Load all task JSON files from the tasks directory."""
    if _task_cache:
        return
    if not _TASKS_DIR.exists():
        logger.warning(f"Tasks directory not found: {_TASKS_DIR}")
        return
    for fp in sorted(_TASKS_DIR.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            task = TaskDefinition(**data)
            _task_cache[task.task_id] = task
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load task file {fp.name}: {e}")


def get_task(task_id: str) -> TaskDefinition:
    """Return a task by its ID. Raises KeyError if not found."""
    _load_tasks()
    if task_id not in _task_cache:
        available = list(_task_cache.keys())
        raise KeyError(f"Task '{task_id}' not found. Available: {available}")
    return _task_cache[task_id]


def list_tasks() -> list[dict[str, str]]:
    """Return a summary list of all available tasks."""
    _load_tasks()
    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
        }
        for t in _task_cache.values()
    ]


def get_all_task_ids() -> list[str]:
    """Return all available task IDs."""
    _load_tasks()
    return list(_task_cache.keys())


def reload_tasks() -> None:
    """Force reload all tasks from disk."""
    _task_cache.clear()
    _load_tasks()
