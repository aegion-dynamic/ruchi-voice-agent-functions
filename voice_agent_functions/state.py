"""Per-session state for the intake + cooking-companion flows.

Each ``VoiceSession`` tracks:

* ``mode``: which agent personality is active (``intake`` or ``cook``).
* ``answers``: collected answers while in intake mode.
* ``current_field``: which intake field is being asked next.
* ``recipe``: ``{recipeName, steps, stepIndex}`` carried by the room
  metadata in cook mode.
* ``paused``: whether the cooking agent is currently waiting.
* ``active_timer``: ``None`` or ``{seconds, label, task}``.
* ``active_issue``: ``None`` or ``{issue, details}``.
* ``events``: ordered list of every state change published by tools
  (``field_saved``, ``step_changed``, ``timer_started``, ...) so the
  connected frontend can render live progress.

The dataclass is intentionally simple and picklable so the same
session can be held in memory by a FastAPI process, persisted to
Redis later, or replayed by tests.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Mode = Literal["intake", "cook"]


# Order of fields the intake agent asks for, from RECIPE_INTAKE_AGENT.md.
INTAKE_FIELDS: tuple[str, ...] = (
    "name",
    "time",
    "servings",
    "level",
    "ingredients",
    "utensils",
    "tips",   # optional — empty string is a valid answer
    "steps",  # last — calls finish_interview after saving
)


@dataclass
class TimerState:
    seconds: int
    label: str
    task: asyncio.Task


@dataclass
class IssueState:
    issue: str  # one of spicy | salty | watery | burnt | bland | other
    details: str


@dataclass
class RecipeContext:
    recipe_name: str
    steps: list[str]
    step_index: int = 0

    def step_text(self) -> str:
        if 0 <= self.step_index < len(self.steps):
            return self.steps[self.step_index]
        return ""

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "RecipeContext":
        steps = list(data.get("steps") or [])
        return cls(
            recipe_name=str(data.get("recipeName", "")),
            steps=steps,
            step_index=int(data.get("stepIndex", 0)),
        )


@dataclass
class VoiceSession:
    session_id: str
    mode: Mode
    answers: dict[str, str] = field(default_factory=dict)
    current_field: str = "name"
    recipe: Optional[RecipeContext] = None
    paused: bool = False
    active_timer: Optional[TimerState] = None
    active_issue: Optional[IssueState] = None
    events: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Intake helpers
    # ------------------------------------------------------------------
    def intake_next_field(self, just_saved: str) -> Optional[str]:
        """Return the field name that comes after ``just_saved`` in
        the canonical intake order, or ``None`` if it was the last one."""
        try:
            idx = INTAKE_FIELDS.index(just_saved)
        except ValueError:
            return None
        if idx + 1 >= len(INTAKE_FIELDS):
            return None
        return INTAKE_FIELDS[idx + 1]

    def intake_field_question(self, field_name: str) -> str:
        """Exact question text for each field (verbatim from the docs)."""
        return {
            "name":       "What's the name of your recipe?",
            "time":       "About how long does it take to cook?",
            "servings":   "How many people does it serve?",
            "level":      "How difficult is it — easy, medium, or hard?",
            "ingredients": "List the ingredients, with approximate quantities.",
            "utensils":   "What utensils do you need? Tell me what you remember.",
            "tips":       "Any quick tips you want to add?",
            "steps":      "Now walk me through all the steps.",
        }.get(field_name, "")

    # ------------------------------------------------------------------
    # Cooking helpers
    # ------------------------------------------------------------------
    def goto_step(self, new_index: int) -> dict[str, Any]:
        assert self.recipe is not None, "recipe context not set"
        clamped = max(0, min(new_index, len(self.recipe.steps) - 1))
        self.recipe.step_index = clamped
        return {
            "type": "step_changed",
            "stepIndex": clamped,
            "stepText": self.recipe.step_text(),
        }


__all__ = [
    "Mode",
    "INTAKE_FIELDS",
    "TimerState",
    "IssueState",
    "RecipeContext",
    "VoiceSession",
]
