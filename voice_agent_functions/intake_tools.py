"""Intake-agent tools: ``save_answer`` and ``finish_interview``.

The intake agent walks a user through 8 fixed fields in order (see
``RECIPE_INTAKE_AGENT.md``). It only has two tools:

* ``save_answer(field, value)`` — store the user's answer for the
  current field, advance the pointer, publish ``field_saved``.
* ``finish_interview()`` — once the last field (``steps``) is saved,
  publish ``interview_complete`` and end the structured part of the
  conversation.

The handler never invents a "current field" — the LLM is the only
thing that decides when an answer is complete enough to call
``save_answer``. The dispatcher enforces that ``field`` matches the
session's current field before letting the call through.
"""
from __future__ import annotations

from typing import Any

from .state import INTAKE_FIELDS, VoiceSession


# Canned acknowledgement phrases the intake agent should rotate through
# right after a successful save_answer. Kept verbatim per the docs.
INTAKE_ACK_PHRASES: tuple[str, ...] = (
    "Got it.",
    "Saved — got it.",
    "Noted, thanks.",
)


# Canned re-ask phrases used when the answer is unclear and save_answer
# is NOT called. Kept verbatim per the docs.
INTAKE_REASK_PHRASES: tuple[str, ...] = (
    "Sorry, one more time?",
    "Could you repeat that?",
    "Didn't quite catch that — could you say it again?",
)


# Canned off-topic redirect — used when the user asks something
# unrelated to the current field, then immediately followed by the
# current field's question again.
INTAKE_REDIRECT_PREFIX = "Anyway, back to your recipe —"


# Closing line spoken after finish_interview fires.
INTAKE_CLOSING_LINE = "That's everything — your recipe's recorded!"


def _validate_intake_field(field: str) -> str | None:
    """Return an error message if ``field`` is not a known intake
    field, otherwise ``None``."""
    if field not in INTAKE_FIELDS:
        return f"Unknown intake field: {field!r}. Must be one of {list(INTAKE_FIELDS)}."
    return None


def save_answer(session: VoiceSession, field: str, value: str) -> dict[str, Any]:
    """Store ``value`` under ``field`` and advance the intake pointer.

    Mirrors the spec from RECIPE_INTAKE_AGENT.md:

    1. Validate ``field`` is the *current* field on the session
       (don't let the LLM skip ahead).
    2. Store ``value`` under ``field``.
    3. Move the internal pointer to the next field.
    4. Publish ``field_saved`` so the app can update its progress UI.
    5. If ``field == "steps"`` (the last one) call ``finish_interview``
       instead of asking another question.

    Returns the ``field_saved`` event dict. Raises ``ValueError`` if
    the field is wrong, so the dispatcher can convert that into a
    Gemini tool ``error`` response.
    """
    err = _validate_intake_field(field)
    if err:
        raise ValueError(err)
    if field != session.current_field:
        raise ValueError(
            f"save_answer called with field={field!r} but current field is "
            f"{session.current_field!r}; refusing to skip ahead."
        )

    # The intake agent spec treats "" / "skip" as valid empty answers
    # for the *optional* ``tips`` field.
    if field == "tips":
        normalized = (value or "").strip()
        if normalized.lower() in {"", "skip", "no", "none", "nothing"}:
            value = ""
        else:
            value = normalized

    session.answers[field] = value

    next_field = session.intake_next_field(field)
    if next_field is None:
        # Last field (``steps``) just got saved — fire the
        # interview_complete chain via finish_interview.
        complete_event = finish_interview(session)
        return {
            "type": "field_saved",
            "field": field,
            "value": value,
            "next_field": None,
            "interview_complete": complete_event,
        }

    session.current_field = next_field
    event = {
        "type": "field_saved",
        "field": field,
        "value": value,
        "next_field": next_field,
    }
    session.events.append(event)
    return event


def finish_interview(session: VoiceSession) -> dict[str, Any]:
    """Publish ``interview_complete`` and freeze the session.

    Per RECIPE_INTAKE_AGENT.md this fires once ``steps`` (the last
    field) has been saved. The handler always includes the ``tips``
    key even when it's an empty string, so the connected frontend
    can rely on a stable schema.
    """
    # Always include ``tips`` even when blank — per spec.
    answers = {f: session.answers.get(f, "") for f in INTAKE_FIELDS}

    event = {
        "type": "interview_complete",
        "answers": answers,
    }
    session.events.append(event)
    # Mark session as done so the runner stops asking questions.
    session.current_field = ""
    return event


__all__ = [
    "INTAKE_ACK_PHRASES",
    "INTAKE_REASK_PHRASES",
    "INTAKE_REDIRECT_PREFIX",
    "INTAKE_CLOSING_LINE",
    "save_answer",
    "finish_interview",
]
