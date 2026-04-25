"""Lightweight calendar environment with conflict handling."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalendarEnv:
    time_slots: list[str]
    scheduled_tasks: dict[str, str] = field(default_factory=dict)
    rejected_tasks: list[str] = field(default_factory=list)

    def schedule_task(self, time_slot: str, task_id: str) -> dict[str, object]:
        """Try to schedule a task in a slot; returns conflict metadata."""
        if time_slot not in self.time_slots:
            return {
                "success": False,
                "conflict": False,
                "message": f"Invalid slot: {time_slot}",
                "slot": time_slot,
            }

        if time_slot in self.scheduled_tasks:
            return {
                "success": False,
                "conflict": True,
                "message": f"Slot {time_slot} is already occupied.",
                "slot": time_slot,
            }

        self.scheduled_tasks[time_slot] = task_id
        return {
            "success": True,
            "conflict": False,
            "message": f"Task {task_id} scheduled at {time_slot}.",
            "slot": time_slot,
        }

    def reschedule_task(self, old_slot: str, new_slot: str) -> dict[str, object]:
        """Move a task between slots when possible."""
        if old_slot not in self.scheduled_tasks:
            return {
                "success": False,
                "conflict": False,
                "message": f"No task found in {old_slot}.",
                "slot": old_slot,
            }

        task_id = self.scheduled_tasks[old_slot]
        schedule_result = self.schedule_task(new_slot, task_id)
        if not schedule_result["success"]:
            return schedule_result

        del self.scheduled_tasks[old_slot]
        return {
            "success": True,
            "conflict": False,
            "message": f"Task {task_id} moved from {old_slot} to {new_slot}.",
            "slot": new_slot,
        }

    def reject_task(self, task_id: str) -> dict[str, object]:
        """Reject a task request when scheduling is not appropriate."""
        self.rejected_tasks.append(task_id)
        return {
            "success": True,
            "conflict": False,
            "message": f"Task {task_id} rejected.",
            "slot": None,
        }

    def first_available_slot(self) -> str | None:
        for slot in self.time_slots:
            if slot not in self.scheduled_tasks:
                return slot
        return None
