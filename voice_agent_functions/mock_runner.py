"""Offline (rule-based) runner for development and tests.

When ``GOOGLE_API_KEY`` is not configured — or in unit tests where we
want to exercise the tool dispatch logic without calling Gemini —
this runner produces canned assistant replies and tool calls based on
simple keyword matching.

It is **not** the production path; the production path is
:class:`runner.GeminiFunctionCallingRunner`. The mock runner exists so
the FastAPI endpoints can return useful responses for the React UI
when running with no API keys set, matching the behavior of the rest
of backend/app.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .dispatcher import dispatch
from .state import VoiceSession
from .cooking_tools import ISSUE_LEAD_INS, NEXT_STEP_LEAD_IN


@dataclass
class MockTurn:
    user_text: str
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------
# Tiny keyword router — kept simple on purpose. The whole point is
# dev-mode demos; the real Gemini loop does the actual understanding.
# ----------------------------------------------------------------------
_NEXT_RE = re.compile(r"\b(next|okay|done|finished|what'?s next|తర్వాత|పూర్తి)\b", re.I)
_PREV_RE = re.compile(r"\b(back|previous|గత|వెనుకకు)\b", re.I)
_GOTO_RE = re.compile(r"\bstep\s*(\d+)\b", re.I)
_TIMER_START_RE = re.compile(r"\b(set|start)\b.*\btimer\b.*?(\d+)\s*(seconds|second|sec|minutes|minute|min)?", re.I)
_TIMER_CANCEL_RE = re.compile(r"\b(cancel|stop)\b.*\btimer\b", re.I)
_PAUSE_RE = re.compile(r"^\s*(pause|wait|hold on|stop)\s*[.!?]?\s*$", re.I)
_RESUME_RE = re.compile(r"^\s*(resume|continue|go ahead|i'?m back)\s*[.!?]?\s*$", re.I)
_RESOLVE_RE = re.compile(r"\b(resolve|fixed|sorted|done with|back to|continue cooking|that worked|all good)\b", re.I)
_FINISH_COOKING_RE = re.compile(r"\b(finish cooking|finish|done cooking|we'?re done|i'?m done)\b", re.I)
_ISSUE_MAP = {
    "spicy": r"\b(spicy|hot|spicier)\b",
    "salty": r"\b(salt|salty|too much salt)\b",
    "watery": r"\b(watery|thin|runny)\b",
    "burnt": r"\b(burnt|burning)\b",
    "bland": r"\b(bland|tasteless|no flavor)\b",
}

# ----------------------------------------------------------------------
# Telugu keyword matching.
#
# `\b` word boundaries are ASCII-oriented and don't behave correctly
# with Telugu script (which uses different word segmentation), so for
# Telugu we use plain substring matching on the lowercased text.
# These are checked BEFORE the English regexes.
# ----------------------------------------------------------------------
_TE_NEXT = ("తర్వాత", "తరువాత", "నెక్స్ట్", "ముందుకు", "ముందు స్టెప్", "అయిపోయింది", "పూర్తయింది")
_TE_PREV = ("వెనుక", "గత", "మునుపటి", "వెనక్కి", "వెనక స్టెప్")
_TE_PAUSE = ("ఆపు", "ఆగండి", "కాసేపు", "వెయిట్", "ఆగు")
_TE_RESUME = ("కొనసాగించు", "కంటిన్యూ", "మళ్ళీ మొదలు", "మళ్లీ మొదలు", "రెజ్యూమ్")
_TE_RESOLVE = ("సరిపోయింది", "బాగుంది", "ఫిక్స్", "సర్దుబాటు", "సరిచేశాను")
_TE_FINISH = ("వంట పూర్త", "వంట అయిపో", "పూర్తి చేయ", "ముగించు")
_TE_ISSUE = {
    "spicy": ("కారం", "మసాలా", "కారంగా"),
    "salty": ("ఉప్పు", "ఉప్పగా"),
    "watery": ("నీరు", "పలుచన", "పలుచగా"),
    "burnt": ("కాలింది", "కరువు", "మాడింది"),
    "bland": ("రుచి లేదు", "రుచిగా లేదు", "చప్పగా"),
}


def _te_has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _fallback_assistant_text(user_text: str) -> str:
    """When no tool fires, return a friendly canned reply so the UI
    has something to render in mock mode."""
    t = user_text.lower().strip()
    if not t:
        return "నేను వింటున్నాను. (డెమో మోడ్)"
    if any(w in t for w in ["hi", "hello", "నమస్కారం"]):
        return "నమస్కారం! నేను రుచి. ఈరోజు ఏం వండుకుందాం? (డెమో మోడ్)"
    return (
        "డెమో మోడ్‌లో ఉన్నాను — GOOGLE_API_KEY సెట్ చేస్తే నిజమైన "
        f"సమాధానం ఇస్తాను. మీరు చెప్పింది: “{user_text[:120]}”"
    )


def _detect_tool_call(user_text: str, mode: str) -> tuple[str | None, dict[str, Any]]:
    """Return ``(tool_name, args)`` if a tool should fire, else ``(None, {})``."""
    if mode == "intake":
        # Mock intake: treat the whole utterance as the answer for the
        # current field. The frontend can drive the field sequence by
        # calling /api/voice-agent/intake/start explicitly.
        return None, {}

    text = user_text.strip()

    # ---- Telugu first (substring match; \b doesn't work with Telugu) ----
    # Order: finish > resolve > pause > next > prev > resume > issue.
    if _te_has(text, _TE_FINISH):
        return "finish_cooking", {}
    if _te_has(text, _TE_RESOLVE):
        return "resolve_issue", {}
    if _te_has(text, _TE_PAUSE) and len(text) < 20:
        return "pause", {}
    if _te_has(text, _TE_NEXT):
        return "next_step", {}
    if _te_has(text, _TE_PREV):
        return "previous_step", {}
    if _te_has(text, _TE_RESUME):
        return "resume", {}
    for cat, words in _TE_ISSUE.items():
        if _te_has(text, words):
            return "report_issue", {"issue": cat, "details": text}

    # ---- English regexes ----
    # Order matters: more-specific phrases first so "done cooking" doesn't
    # fall through to the generic _NEXT_RE.
    if _RESUME_RE.match(text):
        return "resume", {}
    if _PAUSE_RE.match(text):
        return "pause", {}
    if _FINISH_COOKING_RE.search(text):
        return "finish_cooking", {}
    if _RESOLVE_RE.search(text):
        return "resolve_issue", {}
    m = _GOTO_RE.search(text)
    if m:
        return "goto_step", {"step": int(m.group(1))}
    if _TIMER_CANCEL_RE.search(text):
        return "cancel_timer", {}
    m = _TIMER_START_RE.search(text)
    if m:
        amount = int(m.group(2))
        unit = (m.group(3) or "seconds").lower()
        seconds = amount * 60 if unit.startswith("min") else amount
        return "start_timer", {"seconds": seconds, "label": "your timer"}
    if _PREV_RE.search(text) and not _NEXT_RE.search(text):
        return "previous_step", {}
    if _NEXT_RE.search(text):
        return "next_step", {}
    for cat, pat in _ISSUE_MAP.items():
        if re.search(pat, text, re.I):
            return "report_issue", {"issue": cat, "details": text}
    return None, {}


# Telugu canned responses keyed by tool name — used when the user's
# input looks like Telugu, so the companion answers in Telugu instead
# of the English canned lines from the PR docs.
_TE_TOOL_REPLY = {
    "next_step": "సరే, తర్వాత స్టెప్.",
    "previous_step": "సరే, మునుపటి స్టెప్.",
    "goto_step": "స్టెప్‌కి వెళ్తున్నాను.",
    "start_timer": "టైమర్ సెట్ చేశాను.",
    "cancel_timer": "టైమర్ రద్దు చేశాను.",
    "pause": "సరే, ఆగుతాను.",
    "resume": "కొనసాగిస్తున్నాను —",
    "report_issue": "సరే, సరిచేద్దాం —",
    "resolve_issue": "బాగుంది — కొనసాగిద్దాం.",
    "finish_cooking": "భోజనం ఆనందించండి!",
    "save_answer": "సరే, సేవ్ చేశాను.",
    "finish_interview": "అంతా పూర్తయింది — మీ రెసిపీ రికార్డ్ అయింది!",
}


def _looks_telugu(text: str) -> bool:
    """True if ``text`` contains Telugu Unicode characters."""
    return any("\u0c00" <= ch <= "\u0c7f" for ch in text)


async def run_mock_turn(session: VoiceSession, user_text: str) -> MockTurn:
    """Run one turn of the mock loop: dispatch the detected tool (if
    any), then return a canned assistant reply.

    If the user wrote/spoke Telugu, reply with a Telugu canned line so
    the companion feels natural; otherwise use the English phrases from
    the PR docs.
    """
    turn = MockTurn(user_text=user_text)
    is_te = _looks_telugu(user_text)
    tool_name, args = _detect_tool_call(user_text, session.mode)
    if tool_name:
        result = dispatch(session, tool_name, args)
        turn.tool_calls.append({"name": tool_name, "args": args})
        turn.tool_results.append(result)
        if "result" in result:
            phrase = result["result"].get("phrase")
            lead = result["result"].get("lead_in")
            if is_te and tool_name in _TE_TOOL_REPLY:
                turn.assistant_text = _TE_TOOL_REPLY[tool_name] + " (డెమో మోడ్)"
                return turn
            if phrase:
                turn.assistant_text = phrase + " (డెమో మోడ్)"
                return turn
            if lead:
                turn.assistant_text = lead + " (డెమో మోడ్)"
                return turn
        elif "error" in result:
            turn.assistant_text = result["error"]
            return turn

    turn.assistant_text = _fallback_assistant_text(user_text)
    return turn


__all__ = ["MockTurn", "run_mock_turn"]
