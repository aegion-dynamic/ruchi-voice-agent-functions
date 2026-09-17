"""Loads system instructions from the bundled prompts/ folder.

Mirrors the small ``load_instructions(name)`` helper added in
ruchi-voice-agent PR #1 (commit 8414c1b) but stays self-contained
so the FastAPI backend can use it without depending on the
realtime LiveKit worker.
"""
from __future__ import annotations

from pathlib import Path

# voice_agent_functions/prompts_loader.py -> voice_agent_functions/prompts/
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_instructions(name: str) -> str:
    """Return the contents of ``prompts/<name>.txt`` with trailing
    whitespace stripped."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


__all__ = ["PROMPTS_DIR", "load_instructions"]
