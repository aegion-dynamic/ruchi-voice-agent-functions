"""Wires the function-calling tools to the Gemini REST API.

Implements the **standard Gemini tool-use loop**:

1. Send the conversation + ``tools`` to ``generateContent``.
2. If the response contains ``functionCall`` parts, dispatch each one
   via :func:`dispatcher.dispatch_function_call`.
3. Append the resulting ``functionResponse`` parts and loop until
   the model returns a plain text reply.

Both agent modes (``intake`` and ``cook``) share the same loop — only
the system prompt and the ``tools`` payload change.

The runner is intentionally transport-agnostic — it doesn't talk to
LiveKit or any audio plugin. It takes/returns plain text turns, so
the existing FastAPI ``/api/chat`` (or a websocket, or a CLI test) can
drive it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .dispatcher import dispatch_function_call, known_tool_names
from .prompts_loader import load_instructions
from .state import RecipeContext, VoiceSession
from .tool_schemas import gemini_tools

logger = logging.getLogger("ruchi.voice_agent_functions")


# Model fallback chain. The first entry is whatever GEMINI_MODEL is set
# to (loaded from .env); the rest are known-good fallbacks in case the
# configured model has been retired.
DEFAULT_MODEL_CHAIN: tuple[str, ...] = (
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
)


@dataclass
class Turn:
    """One model turn (user input + assistant reply + tool calls/responses)."""

    user_text: str
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _function_response(
    call_id: str | None,
    name: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a ``functionResponse`` part for Gemini.

    * ``id`` must echo the ``functionCall.id`` so the model can match
      the response to the call.
    * The response body is ``{"result": ...}`` on success or
      ``{"error": ...}`` on failure.
    """
    fr: dict[str, Any] = {"name": name}
    if call_id:
        fr["id"] = call_id
    if error is not None:
        fr["response"] = {"error": error}
    else:
        fr["response"] = {"result": result or {}}
    return {"functionResponse": fr}


@dataclass
class RunnerConfig:
    api_key: str = ""
    model_chain: tuple[str, ...] = DEFAULT_MODEL_CHAIN
    max_tool_iters: int = 5
    temperature: float = 0.7
    max_output_tokens: int = 256
    request_timeout: float = 30.0


# ============================================================
# Session manager
# ============================================================
class SessionManager:
    """In-memory store of :class:`VoiceSession` objects keyed by id."""

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}

    def create(
        self,
        mode: Literal["intake", "cook"],
        session_id: str | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> VoiceSession:
        sid = session_id or uuid.uuid4().hex
        recipe_ctx = (
            RecipeContext.from_metadata(recipe) if recipe else None
        )
        session = VoiceSession(
            session_id=sid,
            mode=mode,
            recipe=recipe_ctx,
        )
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> VoiceSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session_id: {session_id!r}") from exc

    def drop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session and session.active_timer is not None:
            session.active_timer.task.cancel()


# ============================================================
# Gemini tool-use loop
# ============================================================
class GeminiFunctionCallingRunner:
    """Runs a Gemini conversation with our 12 function-calling tools."""

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig(api_key=os.getenv("GOOGLE_API_KEY", ""))
        self._client = httpx.AsyncClient(timeout=self.config.request_timeout)
        self._history: dict[str, list[dict[str, Any]]] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------------------------------------------------------- prompts
    def _system_prompt(self, mode: str) -> str:
        if mode == "intake":
            return load_instructions("intake")
        # cook mode: combine the cooking-companion overlay with the
        # base Ruchi personality prompt (the one the PR extracted to
        # prompts/cooking.txt).
        overlay = load_instructions("cooking_companion")
        base = load_instructions("cooking")
        return f"{base}\n\n---\n\n{overlay}"

    # ---------------------------------------------------------- gemini
    def _build_payload(
        self,
        session: VoiceSession,
        history: list[dict[str, Any]],
        user_text: str | None,
    ) -> dict[str, Any]:
        contents = list(history)
        if user_text is not None:
            contents.append({"role": "user", "parts": [{"text": user_text}]})

        return {
            "system_instruction": {"parts": [{"text": self._system_prompt(session.mode)}]},
            "contents": contents,
            "tools": gemini_tools(session.mode),
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }

    async def _call_gemini(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_err = ""
        for model in self.config.model_chain:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{model}:generateContent?key={self.config.api_key}"
            )
            try:
                r = await self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_err = f"network error: {exc}"
                continue
            if r.status_code == 404:
                last_err = f"model {model} not found"
                continue
            if r.status_code in (400, 401, 403):
                raise RuntimeError(f"Gemini {r.status_code}: {r.text[:400]}")
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_err = f"Gemini {r.status_code}: {r.text[:300]}"
                continue
            data = r.json()
            if data.get("candidates"):
                return data
            last_err = f"empty Gemini response: {str(data)[:200]}"
        raise RuntimeError(f"All Gemini models failed: {last_err}")

    # ---------------------------------------------------------- main loop
    async def run_turn(
        self,
        session: VoiceSession,
        user_text: str,
    ) -> Turn:
        """Process one user utterance through the function-calling loop.

        Returns a :class:`Turn` describing the assistant reply and any
        tools that fired. ``session.events`` is appended to as a
        side-effect.
        """
        history = self._history.setdefault(session.session_id, [])
        turn = Turn(user_text=user_text)

        history.append({"role": "user", "parts": [{"text": user_text}]})

        for _ in range(self.config.max_tool_iters):
            payload = self._build_payload(session, history, user_text=None)
            try:
                data = await self._call_gemini(payload)
            except Exception as exc:
                turn.error = str(exc)
                logger.exception("Gemini call failed")
                break

            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )

            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            text_chunks = [p["text"] for p in parts if "text" in p]
            text = "".join(text_chunks).strip()

            if function_calls:
                # Append the model turn with its functionCall parts
                # (including thoughtSignature, which Gemini requires us
                # to echo back verbatim).
                history.append({"role": "model", "parts": parts})
                # Dispatch each call, keeping the functionCall id so the
                # functionResponse can be matched to it.
                response_parts: list[dict[str, Any]] = []
                for fc in function_calls:
                    name = fc.get("name", "")
                    call_id = fc.get("id")
                    if name not in known_tool_names(session.mode):
                        # Defensive — the schema should prevent this, but
                        # if Gemini ever returns an out-of-scope tool we
                        # want a clean error back rather than a crash.
                        response_parts.append(
                            _function_response(
                                call_id, name,
                                error=f"tool {name!r} not in scope for mode={session.mode!r}",
                            )
                        )
                        turn.tool_calls.append({"name": name, "args": fc.get("args", {})})
                        continue
                    result = dispatch_function_call(session, fc)
                    response_parts.append(_function_response(call_id, name, result=result))
                    turn.tool_calls.append({"name": name, "args": fc.get("args", {})})
                    turn.tool_results.append(result)

                # IMPORTANT: this Gemini API rejects role="function";
                # functionResponse parts go in a role="user" turn.
                history.append({"role": "user", "parts": response_parts})
                # Loop back to let the model turn the tool outputs into
                # a final text reply (or call another tool).
                continue

            # Plain text reply — we're done.
            history.append({"role": "model", "parts": [{"text": text}]})
            turn.assistant_text = text
            return turn

        # Fell out of the loop without a final text — surface what we have.
        turn.error = turn.error or "model loop ended without a text reply"
        return turn


# ============================================================
# Convenience: one-shot helper
# ============================================================
async def run_once(
    mode: Literal["intake", "cook"],
    user_text: str,
    *,
    session: VoiceSession | None = None,
    recipe: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> tuple[VoiceSession, Turn]:
    """Run a single user turn end-to-end and return ``(session, turn)``.

    Useful for tests and CLI one-shots. Use a :class:`SessionManager` +
    a long-lived :class:`GeminiFunctionCallingRunner` for production.
    """
    cfg = RunnerConfig(api_key=api_key or os.getenv("GOOGLE_API_KEY", ""))
    runner = GeminiFunctionCallingRunner(cfg)
    try:
        if session is None:
            session = VoiceSession(
                session_id=uuid.uuid4().hex,
                mode=mode,
                recipe=RecipeContext.from_metadata(recipe) if recipe else None,
            )
        turn = await runner.run_turn(session, user_text)
        return session, turn
    finally:
        await runner.aclose()


__all__ = [
    "Turn",
    "RunnerConfig",
    "SessionManager",
    "GeminiFunctionCallingRunner",
    "run_once",
]


# Silence an unused-import warning from the type-check side of the loop
# above when asyncio isn't used elsewhere in this module.
_ = asyncio
