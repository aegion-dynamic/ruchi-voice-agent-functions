"""FastAPI router exposing the function-calling flows over HTTP.

Mounts as ``/api/voice-agent/...`` so the React frontend can drive the
same two-agent design that the LiveKit worker implements.

Endpoints:

* ``POST /sessions``                           — create a new session
  ``{"mode": "intake" | "cook", "recipe"?: {...}}``.
* ``GET  /sessions/{sid}``                    — fetch session state +
  recent events.
* ``DELETE /sessions/{sid}``                  — drop a session.
* ``POST /sessions/{sid}/turn``               — run one user utterance
  through the function-calling loop.
* ``GET  /tools?mode=intake|cook``            — list the function-
  calling schemas (useful for the frontend to render "what Ruchi can
  do").
* ``POST /sessions/{sid}/save_answer``        — explicit shortcut
  used by the intake frontend's "save" button.

When ``GOOGLE_API_KEY`` is not set, every endpoint transparently uses
the rule-based mock runner so the UI works out-of-the-box for
development, matching the rest of ``backend/app.py``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import transcript
from .dispatcher import known_tool_names
from .mock_runner import run_mock_turn
from .runner import GeminiFunctionCallingRunner, RunnerConfig, SessionManager, Turn
from .state import RecipeContext
from .tool_schemas import gemini_tools
from .prompts_loader import load_instructions

logger = logging.getLogger("ruchi.voice_agent_functions.routes")


router = APIRouter(prefix="/api/voice-agent", tags=["voice-agent-functions"])

_sessions = SessionManager()
_runner: GeminiFunctionCallingRunner | None = None


def _get_runner() -> GeminiFunctionCallingRunner:
    """Lazy-build the singleton Gemini runner. If the API key is
    missing we still create the runner — it just won't be used (mock
    runner takes over)."""
    global _runner
    if _runner is None:
        _runner = GeminiFunctionCallingRunner(
            RunnerConfig(api_key=os.getenv("GOOGLE_API_KEY", ""))
        )
    return _runner


# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    mode: Literal["intake", "cook"]
    session_id: str | None = None
    recipe: dict[str, Any] | None = None


class SessionStateResponse(BaseModel):
    session_id: str
    mode: str
    current_field: str
    answers: dict[str, str]
    paused: bool
    active_timer: dict[str, Any] | None = None
    active_issue: dict[str, Any] | None = None
    recipe: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class TurnRequest(BaseModel):
    text: str


class TurnResponse(BaseModel):
    session_id: str
    user_text: str
    assistant_text: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    mock: bool = False
    error: str | None = None


class SaveAnswerRequest(BaseModel):
    field: str
    value: str


class CallToolRequest(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


@router.post("/sessions/{session_id}/call")
async def call_tool(session_id: str, req: CallToolRequest):
    """Invoke a tool directly, bypassing the LLM.

    Used by the demo's function dashboard so pressing ``start_timer``
    fires exactly ``start_timer`` with the given args — no reliance on
    the model interpreting a synthetic utterance.
    """
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    from .dispatcher import dispatch
    result = dispatch(session, req.name, req.args)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    transcript.log_note(
        session_id,
        f"tool fired directly: `{req.name}` {req.args or ''}",
    )
    return {
        "ok": True,
        "name": req.name,
        "result": result["result"],
        "events": list(session.events),
        "session": _session_to_response(session),
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@router.post("/sessions", response_model=SessionStateResponse)
async def create_session(req: CreateSessionRequest):
    if req.mode not in ("intake", "cook"):
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode!r}")
    if req.mode == "cook" and not req.recipe:
        raise HTTPException(
            status_code=400,
            detail=(
                "cook mode requires a recipe context "
                "{recipeName, steps[], stepIndex} — pass it as 'recipe'."
            ),
        )
    session = _sessions.create(
        mode=req.mode,
        session_id=req.session_id,
        recipe=req.recipe,
    )
    return _session_to_response(session)


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _session_to_response(session)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    _sessions.drop(session_id)
    return {"ok": True}


class SetRecipeRequest(BaseModel):
    recipe: dict[str, Any]


@router.put("/sessions/{session_id}/recipe", response_model=SessionStateResponse)
async def set_recipe(session_id: str, req: SetRecipeRequest):
    """Replace the recipe context mid-session.

    Used when the user names a dish ("I want to cook dosa") — the
    frontend swaps the recipe panel and calls this so the step
    navigation tools operate on the new steps.
    """
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    recipe = req.recipe or {}
    if not recipe.get("steps"):
        raise HTTPException(status_code=400, detail="recipe.steps[] is required")
    session.recipe = RecipeContext.from_metadata(recipe)
    session.events.append(
        {
            "type": "recipe_changed",
            "recipeName": session.recipe.recipe_name,
            "steps": list(session.recipe.steps),
            "stepIndex": session.recipe.step_index,
        }
    )
    return _session_to_response(session)


@router.post("/sessions/{session_id}/turn", response_model=TurnResponse)
async def run_turn(session_id: str, req: TurnRequest):
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    use_mock = not os.getenv("GOOGLE_API_KEY")
    if use_mock:
        from .mock_runner import MockTurn  # local import to avoid cycles
        mt: MockTurn = await run_mock_turn(session, req.text)
        transcript.log_turn(
            session_id=session_id,
            mode=session.mode,
            user_text=mt.user_text,
            assistant_text=mt.assistant_text,
            tool_calls=mt.tool_calls,
            events=list(session.events),
            provider="mock",
        )
        return TurnResponse(
            session_id=session_id,
            user_text=mt.user_text,
            assistant_text=mt.assistant_text,
            tool_calls=mt.tool_calls,
            tool_results=mt.tool_results,
            events=list(session.events),
            mock=True,
        )

    runner = _get_runner()
    turn: Turn = await runner.run_turn(session, req.text)
    transcript.log_turn(
        session_id=session_id,
        mode=session.mode,
        user_text=turn.user_text,
        assistant_text=turn.assistant_text,
        tool_calls=turn.tool_calls,
        events=list(session.events),
        provider=os.getenv("GEMINI_MODEL", "gemini"),
    )
    return TurnResponse(
        session_id=session_id,
        user_text=turn.user_text,
        assistant_text=turn.assistant_text,
        tool_calls=turn.tool_calls,
        tool_results=turn.tool_results,
        events=list(session.events),
        mock=False,
        error=turn.error,
    )


@router.post("/sessions/{session_id}/save_answer")
async def explicit_save_answer(session_id: str, req: SaveAnswerRequest):
    """Direct frontend-driven call (e.g. tapping a 'save' button after
    typing an answer in the intake form). Identical contract to the
    ``save_answer`` function tool — exposed as its own endpoint so the
    UI doesn't need to round-trip through the LLM just to advance."""
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if session.mode != "intake":
        raise HTTPException(
            status_code=400,
            detail="explicit /save_answer only valid in intake mode",
        )
    from .dispatcher import dispatch
    result = dispatch(session, "save_answer", {"field": req.field, "value": req.value})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"ok": True, "result": result["result"], "session": _session_to_response(session)}


@router.get("/tools")
async def list_tools(mode: Literal["intake", "cook"]):
    """Return the Gemini function-calling schemas for the given mode,
    plus the names of the Python handlers they'll dispatch to."""
    return {
        "mode": mode,
        "tools": gemini_tools(mode),
        "tool_names": sorted(known_tool_names(mode)),
    }


@router.get("/prompts")
async def list_prompts():
    """Return the raw system-prompt text for both modes (handy for
    debugging what Ruchi actually sees)."""
    return {
        "base": load_instructions("cooking"),
        "intake": load_instructions("intake"),
        "cooking_companion": load_instructions("cooking_companion"),
    }


# ----------------------------------------------------------------------
# Transcript — the full user ↔ Ruchi conversation in text form
# ----------------------------------------------------------------------
@router.get("/transcript")
async def get_transcript():
    """Return the whole conversation transcript as Markdown text."""
    from fastapi.responses import PlainTextResponse
    text = transcript.read_transcript()
    if not text:
        text = "# Ruchi — Conversation Transcript\n\n_(no conversation yet)_\n"
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


@router.get("/transcript/files")
async def get_transcript_files():
    """List the transcript files on disk."""
    return {
        "dir": str(transcript.transcript_dir()),
        "files": transcript.transcript_files(),
    }


@router.delete("/transcript")
async def delete_transcript():
    """Clear all transcript files."""
    transcript.clear_transcript()
    return {"ok": True}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _session_to_response(session) -> SessionStateResponse:
    timer = None
    if session.active_timer is not None:
        timer = {
            "seconds": session.active_timer.seconds,
            "label": session.active_timer.label,
        }
    issue = None
    if session.active_issue is not None:
        issue = {
            "issue": session.active_issue.issue,
            "details": session.active_issue.details,
        }
    recipe = None
    if session.recipe is not None:
        recipe = {
            "recipeName": session.recipe.recipe_name,
            "steps": list(session.recipe.steps),
            "stepIndex": session.recipe.step_index,
        }
    return SessionStateResponse(
        session_id=session.session_id,
        mode=session.mode,
        current_field=session.current_field,
        answers=dict(session.answers),
        paused=session.paused,
        active_timer=timer,
        active_issue=issue,
        recipe=recipe,
        events=list(session.events),
    )


__all__ = ["router"]
