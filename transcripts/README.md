# Ruchi Conversation Transcripts

Every exchange between you and Ruchi is appended here automatically, in
plain text (Markdown).

| File | Contents |
|---|---|
| `conversation.md` | The full append-only log across all sessions |
| `YYYY-MM-DD.md` | Per-day logs |

Each entry looks like:

```markdown
### 2026-09-18 00:15:40 · session `5ba8598c` · cook

**You:** నాకు మసాలా దోసె చేయాలి

**Ruchi:** సరే! మసాలా దోసె చేద్దాం. ముందుగా బియ్యం, మినప్పప్పు నాలుగు గంటలు నానబెట్టండి.

`tools: recipe_changed()`
```

## How it's written

Logging happens server-side in `voice_agent_functions/transcript.py`,
called from the `/turn` and `/call` endpoints. Nothing is sent to any
third party — it's a local file.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_ENABLED` | `1` | set `0` to disable logging |
| `TRANSCRIPT_DIR` | `./transcripts` | output directory |

## Viewing / clearing

```bash
# View the transcript
curl http://127.0.0.1:8000/api/voice-agent/transcript

# List transcript files
curl http://127.0.0.1:8000/api/voice-agent/transcript/files

# Clear all transcripts
curl -X DELETE http://127.0.0.1:8000/api/voice-agent/transcript

# Or just read the file
cat transcripts/conversation.md
```
