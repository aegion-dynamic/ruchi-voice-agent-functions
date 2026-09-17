"""Function-calling tools for Ruchi.

Two agent modes, matching the design from ruchi-voice-agent PR #1:

- **Intake mode** (scripted recipe recording): ``save_answer``,
  ``finish_interview``.
- **Cooking mode** (freeform cooking companion): ``next_step``,
  ``previous_step``, ``goto_step``, ``start_timer``, ``cancel_timer``,
  ``pause``, ``resume``, ``report_issue``, ``resolve_issue``,
  ``finish_cooking``.

The tools are exposed to the LLM as Gemini function-calling
declarations (see ``tool_schemas.py``); the LLM picks which one to
invoke and with which arguments; ``dispatcher.py`` then routes the
call to the matching Python handler.
"""
