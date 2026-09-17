"""Standalone FastAPI entry point for voice_agent_functions.

Lets you run the demo + 12 function-calling tools from this folder
without needing the full ruchi-basicfullstack backend:

    cd ~/Projects/voice_agent_functions
    ./run.sh
    # then open http://127.0.0.1:8000/demo/

Serving the demo over http://127.0.0.1 (a "secure context") is
important — browsers only allow microphone access and Web Speech
recognition on localhost or HTTPS. Opening demo/index.html directly
as a file:// URL blocks the mic.

Endpoints:
  GET  /                                  server info
  GET  /api/health                        status
  GET  /api/voice/capabilities            which server-side voice bits exist
  POST /api/voice/tts   {text, voice?}    → MP3 (edge-tts Telugu neural)
  POST /api/voice/stt   multipart file    → {transcript} (Vosk Telugu, offline)
  GET  /api/voice/voices                  list available Telugu voices
  + all /api/voice-agent/* tool endpoints
  + /demo/                                the interactive demo (static)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env BEFORE importing modules that read env vars at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from voice_agent_functions import server_voice
from voice_agent_functions.routes import router as voice_agent_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("ruchi.standalone")

_HERE = Path(__file__).resolve().parent
DEMO_DIR = _HERE / "demo"

app = FastAPI(
    title="Ruchi — voice_agent_functions standalone",
    version="2.1.0",
    description=(
        "Standalone FastAPI server exposing the 12 function-calling tools "
        "from ruchi-voice-agent PR #1, plus free Telugu TTS (edge-tts) and "
        "offline Telugu STT (Vosk). Open /demo/ to interact."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_agent_router)


# ---------------------------------------------------------------- models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


# ---------------------------------------------------------------- info
@app.get("/api/health")
async def health():
    caps = server_voice.capabilities()
    return {
        "status": "ok",
        "name": "voice_agent_functions standalone",
        "voice": caps,
        "demo": "/demo/",
        "endpoints": [
            "/api/voice-agent/sessions",
            "/api/voice-agent/sessions/{sid}",
            "/api/voice-agent/sessions/{sid}/turn",
            "/api/voice-agent/sessions/{sid}/call",
            "/api/voice-agent/sessions/{sid}/recipe",
            "/api/voice-agent/sessions/{sid}/save_answer",
            "/api/voice-agent/tools",
            "/api/voice-agent/prompts",
            "/api/voice-agent/transcript",
            "/api/voice/tts",
            "/api/voice/stt",
            "/api/voice/voices",
            "/api/voice/capabilities",
            "/demo/",
            "/docs",
        ],
    }


@app.get("/")
async def root():
    return {
        "name": "Ruchi — voice_agent_functions standalone",
        "demo": "http://127.0.0.1:8000/demo/  (open this, not the file:// path)",
        "docs": "/docs",
        "health": "/api/health",
    }


# ---------------------------------------------------------------- voice
@app.get("/api/voice/capabilities")
async def voice_capabilities():
    return server_voice.capabilities()


@app.get("/api/voice/voices")
async def voice_voices():
    return {"voices": list(server_voice.TELUGU_VOICES), "default": server_voice.DEFAULT_TTS_VOICE}


@app.post("/api/voice/tts")
async def voice_tts(req: TTSRequest):
    """Synthesize Telugu speech.

    Preference order:
      1. ElevenLabs (``ELEVEN_MODEL_ID``, default ``eleven_v3_conversational``)
      2. edge-tts (free, no key)
      3. browser speechSynthesis (frontend fallback)
    Returns audio/mpeg, or a JSON ``{fallback}`` object.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    # 1) ElevenLabs
    if server_voice.elevenlabs_configured():
        try:
            audio = await server_voice.synthesize_elevenlabs(req.text, req.voice)
            return Response(
                content=audio,
                media_type="audio/mpeg",
                headers={"X-TTS-Provider": "elevenlabs"},
            )
        except Exception as exc:
            log.warning("ElevenLabs TTS failed (%s) — falling back to edge-tts", exc)

    # 2) edge-tts
    if server_voice.capabilities().get("edge_tts"):
        try:
            audio = await server_voice.synthesize_edge_tts(
                req.text, req.voice or server_voice.DEFAULT_TTS_VOICE
            )
            return Response(
                content=audio,
                media_type="audio/mpeg",
                headers={"X-TTS-Provider": "edge-tts"},
            )
        except Exception as exc:
            log.warning("edge-tts failed: %s", exc)

    # 3) Browser
    return JSONResponse(
        status_code=200,
        content={"fallback": "browser", "reason": "no server TTS available", "text": req.text},
    )


@app.post("/api/voice/stt")
async def voice_stt(file: UploadFile = File(...)):
    """Transcribe Telugu audio.

    Preference order:
      1. Sarvam saaras:v3 (``SARVAM_API_KEY``)
      2. Vosk (offline, no key)
      3. browser SpeechRecognition (frontend fallback)
    Accepts webm/ogg/wav/mp3.
    """
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio file")
    fname = file.filename or "audio.webm"
    ctype = file.content_type or "audio/webm"

    # 1) Sarvam
    if server_voice.sarvam_configured():
        try:
            transcript = await server_voice.transcribe_sarvam(audio, fname, ctype)
            return {"transcript": transcript, "provider": "sarvam"}
        except Exception as exc:
            log.warning("Sarvam STT failed (%s) — falling back to Vosk", exc)

    # 2) Vosk
    if server_voice.capabilities().get("vosk") and server_voice.capabilities().get("vosk_model"):
        try:
            transcript = server_voice.transcribe_vosk(audio, fname)
            return {"transcript": transcript, "provider": "vosk"}
        except Exception as exc:
            log.warning("Vosk STT failed: %s", exc)

    # 3) Browser
    return JSONResponse(
        status_code=200,
        content={"transcript": "", "fallback": "browser", "reason": "no server STT available"},
    )


# ---------------------------------------------------------------- demo
if DEMO_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")
else:  # pragma: no cover
    log.warning("demo/ folder not found at %s", DEMO_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
