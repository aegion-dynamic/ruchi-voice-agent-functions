"""Gemini ``tools[].functionDeclarations`` for both agent modes.

Each entry follows the
`Gemini function-calling schema <https://ai.google.dev/gemini-api/docs/function-calling>`_:

* ``name`` — tool name (must match the Python handler in
  ``intake_tools.py`` / ``cooking_tools.py``).
* ``description`` — what the tool does and *when* to call it, as
  written in the PR docs.
* ``parameters`` — JSON Schema describing the arguments.

The descriptions are deliberately written the same way the docs
spell out the *trigger conditions* — that's the part the LLM uses
to decide whether to invoke the tool.
"""
from __future__ import annotations

from typing import Any


# ----------------------------------------------------------------------
# Intake mode — RECIPE_INTAKE_AGENT.md
# ----------------------------------------------------------------------
INTAKE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "save_answer",
        "description": (
            "Call ONLY when you're confident the user's answer to the CURRENT "
            "intake field is complete and clear. This is the 'advance' decision "
            "— calling this tool is what moves the interview forward. If you're "
            "not confident, do NOT call any tool; just ask the user to clarify "
            "(using one of the fixed re-ask phrases)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["name", "time", "servings", "level",
                             "ingredients", "utensils", "tips", "steps"],
                    "description": (
                        "The intake field being saved. Must equal the session's "
                        "current field; the handler will reject mismatches."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "The user's answer for this field. For 'tips' an empty "
                        "string is valid when the user says they have nothing "
                        "to add."
                    ),
                },
            },
            "required": ["field", "value"],
        },
    },
    {
        "name": "finish_interview",
        "description": (
            "Call once the 'steps' field (the last intake field) has been saved. "
            "Publishes the final interview_complete event with every collected "
            "answer and ends the structured part of the conversation."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


# ----------------------------------------------------------------------
# Cooking-companion mode — COOKING_AGENT.md / STEP_NAVIGATION.md / TIMER.md /
# PAUSE_RESUME.md / FIX_MY_DISH.md
# ----------------------------------------------------------------------
COOKING_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "next_step",
        "description": (
            "Call when the user indicates they're ready to move forward — "
            "'okay', 'next', 'done', 'what's next', 'I finished that'. "
            "Advances stepIndex by 1."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "previous_step",
        "description": (
            "Call when the user wants to go back — 'wait, go back', 'what was "
            "the last step again', 'go back a step'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "goto_step",
        "description": (
            "Call when the user names a specific step number directly — "
            "'go to step 4', 'skip to the end', 'take me to step 2'. Steps "
            "are 1-indexed in user speech; the handler converts to 0-indexed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The 1-indexed step number the user asked for.",
                },
            },
            "required": ["step"],
        },
    },
    {
        "name": "start_timer",
        "description": (
            "Call ONLY when the user explicitly asks to start/set a timer — "
            "not just when they mention a duration in passing ('this takes "
            "about 20 minutes' is NOT a request to start a timer; 'set a "
            "timer for 20 minutes' IS)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How long the timer should run, in seconds.",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short human description of what the timer is for "
                        "(e.g. 'the rice', 'on medium flame'). Pull from "
                        "context if the user gave one, otherwise something "
                        "generic like 'your timer' is fine."
                    ),
                },
            },
            "required": ["seconds", "label"],
        },
    },
    {
        "name": "cancel_timer",
        "description": (
            "Call when the user asks to stop/cancel a running timer — "
            "'cancel that timer', 'never mind the timer'. If no timer is "
            "running the handler returns a 'no timer running' response."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "pause",
        "description": (
            "Call ONLY when the user is directly instructing Ruchi to stop "
            "talking and wait — NOT when 'pause'/'wait'/'stop' comes up while "
            "they're talking about the cooking itself (e.g. 'should I pause "
            "the stirring?' is a technique question, not a command)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "resume",
        "description": (
            "Call when a paused user tells Ruchi to continue — 'okay, go "
            "ahead', 'continue', 'I'm back, keep going', 'alright, what's "
            "next'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "report_issue",
        "description": (
            "Call when the user describes a problem with the dish — 'it's too "
            "spicy', 'this is burning', etc. Picks one of the 6 fixed issue "
            "categories; use 'other' when none of the first five match. "
            "DO NOT call this for general cooking questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "issue": {
                    "type": "string",
                    "enum": ["spicy", "salty", "watery", "burnt", "bland", "other"],
                    "description": "One of the 6 fixed issue category ids.",
                },
                "details": {
                    "type": "string",
                    "description": (
                        "Short free-text capture of what the user actually "
                        "said, especially important when issue == 'other'."
                    ),
                },
            },
            "required": ["issue", "details"],
        },
    },
    {
        "name": "resolve_issue",
        "description": (
            "Call once the user confirms a previously reported issue is "
            "fixed (or they've moved on) and wants to get back to normal "
            "cooking flow."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "finish_cooking",
        "description": (
            "Call when the user indicates they're done cooking or has "
            "confirmed the last step is complete and doesn't want to "
            "continue. Ends the session."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


# ----------------------------------------------------------------------
# Aggregates — what gets sent to Gemini
# ----------------------------------------------------------------------
def gemini_tools(mode: str) -> list[dict[str, Any]]:
    """Return the ``tools`` payload to send to Gemini for ``mode``
    (``"intake"`` or ``"cook"``)."""
    declarations = (
        INTAKE_TOOL_SCHEMAS if mode == "intake" else COOKING_TOOL_SCHEMAS
    )
    return [{"functionDeclarations": declarations}]


__all__ = [
    "INTAKE_TOOL_SCHEMAS",
    "COOKING_TOOL_SCHEMAS",
    "gemini_tools",
]
