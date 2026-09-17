/* ============================================================
   Ruchi — Telugu Cooking Companion
   Live voice assistant + all 12 function-calling tools
   ============================================================ */

const API_BASE =
  (location.search.match(/api=([^&]+)/) || [, 'http://localhost:8000'])[1];

// Recipe library. When the user names one of these dishes the recipe
// panel swaps instantly and the session's recipe context is updated so
// the step-navigation tools operate on the new steps.
const RECIPES = {
  biryani: {
    recipeName: 'Paneer Biryani',
    steps: [
      'Soak the rice for 30 minutes.',
      'Marinate the paneer in yogurt and spices.',
      'Layer rice and paneer in a heavy pot.',
      'Cook on low heat for 20 minutes.',
    ],
  },
  dosa: {
    recipeName: 'Masala Dosa',
    steps: [
      'Soak rice and urad dal for 4 hours.',
      'Grind to a smooth batter.',
      'Ferment the batter overnight.',
      'Spread thinly on a hot tawa.',
      'Add potato masala and fold.',
    ],
  },
  poha: {
    recipeName: 'Poha',
    steps: [
      'Rinse the poha and drain well.',
      'Heat oil and splutter mustard seeds.',
      'Add onions, curry leaves and turmeric.',
      'Mix in the poha and cook 2 minutes.',
      'Finish with lemon and coriander.',
    ],
  },
  upma: {
    recipeName: 'Upma',
    steps: [
      'Dry roast the rava until fragrant.',
      'Splutter mustard, add onions and green chilli.',
      'Add water, salt and bring to a boil.',
      'Stir in the rava and cook until thick.',
      'Garnish with coriander and serve.',
    ],
  },
  'paneer butter masala': {
    recipeName: 'Paneer Butter Masala',
    steps: [
      'Soak cashews and blend to a paste.',
      'Sauté onions, tomato and spices.',
      'Blend into a smooth gravy.',
      'Add butter, cream and cashew paste.',
      'Fold in paneer cubes and simmer.',
    ],
  },
  dal: {
    recipeName: 'Dal Tadka',
    steps: [
      'Wash and pressure cook the dal.',
      'Prepare the tadka with ghee and cumin.',
      'Add onions, tomato and turmeric.',
      'Mix the tadka into the dal.',
      'Finish with coriander and lemon.',
    ],
  },
  pulao: {
    recipeName: 'Vegetable Pulao',
    steps: [
      'Rinse and soak the rice 20 minutes.',
      'Sauté whole spices and onions.',
      'Add vegetables and cook 3 minutes.',
      'Add rice and water, bring to a boil.',
      'Cover and cook on low for 15 minutes.',
    ],
  },
  pongal: {
    recipeName: 'Ven Pongal',
    steps: [
      'Roast moong dal until golden.',
      'Cook rice and dal together until soft.',
      'Prepare a tadka of ghee, cumin and pepper.',
      'Pour the tadka over the pongal.',
      'Serve hot with sambar and chutney.',
    ],
  },
};

// Telugu / English dish-name → library key
const DISH_ALIASES = {
  'biryani': 'biryani', 'biriyani': 'biryani', 'బిర్యానీ': 'biryani',
  'dosa': 'dosa', 'దోసె': 'dosa', 'దోశ': 'dosa',
  'poha': 'poha', 'పోహా': 'poha', 'అటుకులు': 'poha',
  'upma': 'upma', 'ఉప్మా': 'upma',
  'paneer butter masala': 'paneer butter masala', 'paneer': 'paneer butter masala',
  'dal': 'dal', 'dal tadka': 'dal', 'పప్పు': 'dal', 'పప్పు చారు': 'dal',
  'pulao': 'pulao', 'pulav': 'pulao', 'పులావ్': 'pulao',
  'pongal': 'pongal', 'పొంగల్': 'pongal',
};

/** Return a recipe object if the user's text names a known dish. */
function detectDish(text) {
  const t = (text || '').toLowerCase();
  // longest alias first so "paneer butter masala" wins over "paneer"
  const keys = Object.keys(DISH_ALIASES).sort((a, b) => b.length - a.length);
  for (const alias of keys) {
    if (t.includes(alias)) return RECIPES[DISH_ALIASES[alias]];
  }
  return null;
}

// Intent words that indicate the user wants to *start* a dish, not
// comment on the one they're cooking.
const DISH_INTENT = [
  'want', 'make', 'cook', 'prepare', 'recipe', 'how to', 'how do i',
  'show me', 'start', 'చేయాలి', 'వండాలి', 'రెసిపీ', 'చేద్దాం',
  'వండుకుందాం', 'కావాలి', 'ఎలా చేయ', 'చెప్పు',
];

function isDishRequest(text) {
  const t = (text || '').toLowerCase();
  return DISH_INTENT.some((w) => t.includes(w));
}

let RECIPE = RECIPES.biryani;   // current recipe (mutable)

const INTAKE_FIELDS = [
  { name: 'name',        question: "What's the name of your recipe?" },
  { name: 'time',        question: 'About how long does it take to cook?' },
  { name: 'servings',    question: 'How many people does it serve?' },
  { name: 'level',       question: 'How difficult is it — easy, medium, or hard?' },
  { name: 'ingredients', question: 'List the ingredients, with approximate quantities.' },
  { name: 'utensils',    question: 'What utensils do you need?' },
  { name: 'tips',        question: 'Any quick tips you want to add? (optional)' },
  { name: 'steps',       question: 'Now walk me through all the steps.' },
];

const state = {
  sessionId: null,
  mode: 'cook',
  pollHandle: null,
  busy: false,
  listening: false,
  recognition: null,
  voices: [],
  caps: null,
  // MediaRecorder path
  recorder: null,
  recordStream: null,
  recordChunks: [],
  recordStart: 0,
  analyser: null,
  audioCtx: null,
  orbState: 'idle',
  // Conversation mode (hands-free loop)
  conversationMode: false,
  paused: false,       // set by the `pause` tool / "stop" intent
  emptyTurns: 0,       // consecutive turns with no speech
  finishRequested: false,
  // Event-driven state
  lastState: null,     // most recent session snapshot
  eventCount: 0,       // how many events we've already applied
  // Live timer
  timerHandle: null,
  timerEndsAt: null,
  timerLabel: null,
};

// ---------- DOM helpers --------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- Server health ------------------------------------------------
async function ping() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) throw new Error('not ok');
    const j = await r.json();
    const pill = $('#server-pill');
    pill.textContent = 'live';
    pill.classList.remove('dead');
    pill.classList.add('live');
  } catch {
    const pill = $('#server-pill');
    pill.textContent = 'offline';
    pill.classList.remove('live');
    pill.classList.add('dead');
  }
}

// ---------- Server voice capabilities ---------------------------------
async function loadCaps() {
  try {
    const r = await fetch(`${API_BASE}/api/voice/capabilities`);
    state.caps = await r.json();
  } catch {
    state.caps = { edge_tts: false, vosk: false, vosk_model: false, ffmpeg: false };
  }
  return state.caps;
}

// ---------- Speech synthesis (edge-tts server → browser fallback) -----
function loadVoices() {
  return new Promise((resolve) => {
    let v = window.speechSynthesis?.getVoices() || [];
    if (v.length) return resolve(v);
    let tries = 0;
    const id = setInterval(() => {
      v = window.speechSynthesis?.getVoices() || [];
      if (v.length || ++tries > 10) { clearInterval(id); resolve(v); }
    }, 200);
  });
}

function pickTeluguVoice() {
  const voices = state.voices || [];
  return (
    voices.find((v) => /te-IN|telugu/i.test(v.lang + v.name)) ||
    voices.find((v) => /hi-IN|hindi|india/i.test(v.lang + v.name)) ||
    voices.find((v) => v.lang?.toLowerCase().startsWith('en')) ||
    null
  );
}

function speakBrowser(text) {
  return new Promise((resolve) => {
    if (!text || !window.speechSynthesis) return resolve();
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const v = pickTeluguVoice();
    if (v) u.voice = v;
    u.lang = v?.lang || 'te-IN';
    u.rate = 0.95;
    u.onstart = () => setOrb('speaking', 'speaking…');
    u.onend = () => { resolve(); };
    u.onerror = () => { resolve(); };
    window.speechSynthesis.speak(u);
  });
}

let _ttsAudio = null;
/** Speak `text`. Resolves when playback finishes (or immediately on failure). */
async function speak(text, { interrupt = true } = {}) {
  if (!text) return;
  setOrb('speaking', 'speaking…');

  // 1) Prefer server-side TTS (ElevenLabs → edge-tts, chosen server-side)
  if (state.caps?.elevenlabs || state.caps?.edge_tts) {
    try {
      if (interrupt && _ttsAudio) { _ttsAudio.pause(); _ttsAudio = null; }
      const r = await fetch(`${API_BASE}/api/voice/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const ct = r.headers.get('content-type') || '';
      if (ct.includes('audio')) {
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        _ttsAudio = audio;
        await new Promise((resolve) => {
          audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
          audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
          audio.play().catch(() => resolve());
        });
        return;
      }
      // JSON fallback response → use browser
    } catch (e) {
      log('error', 'server TTS failed', String(e).slice(0, 120));
    }
  }

  // 2) Browser fallback
  await speakBrowser(text);
}

// ---------- Speech recognition ----------------------------------------
// Primary: MediaRecorder → POST /api/voice/stt (Vosk Telugu, offline).
// This works in every browser over localhost, unlike webkitSpeechRecognition
// which Brave disables and which needs a secure context anyway.
// Secondary: browser SpeechRecognition if server STT is unavailable.

async function startListening() {
  if (state.listening) return;
  if (state.caps?.vosk && state.caps?.vosk_model && state.caps?.ffmpeg) {
    return startServerListening();
  }
  return startBrowserListening();
}

// ---- server path (MediaRecorder + Vosk) ----
async function startServerListening() {
  let stream;
  try {
    // echoCancellation stops Ruchi's own voice from being re-recorded,
    // which is essential for hands-free back-and-forth.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (e) {
    log('error', 'microphone blocked', 'Allow mic access for this site, then retry.');
    pushBubble('system', '⚠ Microphone blocked. Click the mic icon in the address bar and allow access, then try again.');
    endConversation();
    return;
  }
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  state.listening = true;
  state.recorder = rec;
  state.recordStream = stream;
  state.recordChunks = chunks;

  // Silence detection — auto-stop ~1.2s after speech ends.
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const src = ac.createMediaStreamSource(stream);
  const analyser = ac.createAnalyser();
  analyser.fftSize = 1024;
  src.connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  state.analyser = analyser;
  state.audioCtx = ac;

  let spoke = false;
  let lastLoud = performance.now();
  const tick = () => {
    if (!state.listening) return;
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i] - 128));
    const loud = peak > 8;
    if (loud) {
      if (!spoke) { state.emptyTurns = 0; }  // real speech detected
      spoke = true;
      lastLoud = performance.now();
    }
    const now = performance.now();
    if (spoke && now - lastLoud > 1200) { stopListening(); return; }
    // No speech at all: give more time in conversation mode.
    const idleLimit = state.conversationMode ? 10000 : 8000;
    if (!spoke && now - state.recordStart > idleLimit) { stopListening(); return; }
    requestAnimationFrame(tick);
  };

  rec.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
  rec.onstop = () => transcribeAndSend();
  rec.start();
  state.recordStart = performance.now();
  setOrb('listening', state.conversationMode ? 'వింటున్నాను… (hands-free)' : 'వింటున్నాను… (speak, auto-stops)');
  updateMicUi();
  updateVoicePill();
  requestAnimationFrame(tick);
}

async function transcribeAndSend() {
  const chunks = state.recordChunks || [];
  const blob = new Blob(chunks, { type: 'audio/webm' });
  // cleanup
  try { state.recordStream?.getTracks().forEach((t) => t.stop()); } catch {}
  try { state.audioCtx?.close(); } catch {}
  state.recorder = null; state.recordStream = null; state.audioCtx = null;

  if (!blob.size) { afterTurn(); return; }

  setOrb('thinking', 'transcribing…');
  try {
    const fd = new FormData();
    fd.append('file', blob, 'voice.webm');
    const r = await fetch(`${API_BASE}/api/voice/stt`, { method: 'POST', body: fd });
    const j = await r.json();
    const text = (j.transcript || '').trim();
    if (text) {
      log('success', 'STT', { transcript: text });
      $('#input-text').value = text;
      await sendMessage(text);
    } else {
      log('system', 'STT', j.reason ? `no speech (${j.reason})` : 'no speech detected');
      // In conversation mode, just listen again instead of stopping.
      if (state.conversationMode) {
        state.emptyTurns = (state.emptyTurns || 0) + 1;
        if (state.emptyTurns >= 3) {
          pushBubble('system', 'No speech heard 3 times — conversation paused. Tap the orb to resume.');
          endConversation();
        } else {
          setOrb('listening', 'still listening…');
          await wait(400);
          startListening();
        }
      } else {
        pushBubble('system', 'No speech detected — try again or type.');
        afterTurn();
      }
    }
  } catch (e) {
    log('error', 'STT request failed', String(e).slice(0, 120));
    afterTurn();
  }
}

// ---- browser path (webkitSpeechRecognition) ----
function startBrowserListening() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    pushBubble('system', '⚠ Voice input unavailable. Install server STT (vosk) or use Chrome. Type instead.');
    return;
  }
  const r = new SR();
  r.lang = 'te-IN';
  r.continuous = false;
  r.interimResults = true;
  r.maxAlternatives = 1;
  r.onstart = () => {
    state.listening = true;
    setOrb('listening', 'వింటున్నాను…');
    $('#mic-btn').classList.add('danger');
    $('#mic-btn').textContent = '⏹';
    updateVoicePill();
  };
  r.onresult = (ev) => {
    let interim = '', final = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const alt = ev.results[i][0];
      if (ev.results[i].isFinal) final += alt.transcript;
      else interim += alt.transcript;
    }
    if (interim) {
      $('#input-text').value = interim;
      setOrb('listening', `వింటున్నాను: "${interim.slice(0, 30)}…"`);
    }
    if (final) {
      $('#input-text').value = final.trim();
      stopListening();
      sendMessage(final.trim());
    }
  };
  r.onerror = (ev) => {
    log('error', 'browser STT error', ev.error || 'unknown');
    if (ev.error === 'network' || ev.error === 'service-not-allowed') {
      pushBubble('system', 'Browser speech recognition unavailable — using server STT is recommended.');
    }
    stopListening();
  };
  r.onend = () => { if (state.listening) stopListening(); };
  try { r.start(); state.recognition = r; }
  catch (e) { log('error', 'STT start failed', e.message); stopListening(); }
}

/** Stop capturing (to transcribe). Does NOT end the conversation. */
function stopListening() {
  if (state.recorder && state.recorder.state !== 'inactive') {
    try { state.recorder.stop(); } catch {}
    state.listening = false;
    updateMicUi();
    return;
  }
  if (state.recognition) {
    try { state.recognition.stop(); } catch {}
    state.recognition = null;
  }
  state.listening = false;
  updateMicUi();
}

/** Called after a turn finishes. Restarts listening if in conversation mode. */
async function afterTurn() {
  if (state.conversationMode && !state.paused) {
    await wait(300);
    if (state.conversationMode && !state.paused) {
      startListening();
      return;
    }
  }
  resetMicUi();
}

function resetMicUi() {
  state.listening = false;
  setOrb('idle', state.conversationMode ? 'conversation ended' : 'tap to talk');
  updateMicUi();
  updateVoicePill();
}

function updateMicUi() {
  const btn = $('#mic-btn');
  const orb = $('#orb');
  if (state.conversationMode) {
    btn.classList.add('danger');
    btn.textContent = '⏹';
    btn.title = 'Stop conversation';
    orb.classList.add('convo');
  } else {
    btn.classList.remove('danger');
    btn.textContent = '🎤';
    btn.title = 'Start conversation';
    orb.classList.remove('convo');
  }
}

// ---------- Conversation mode (hands-free loop) ------------------------
async function startConversation() {
  state.conversationMode = true;
  state.paused = false;
  state.emptyTurns = 0;
  updateMicUi();
  pushBubble('system', '🎙 Conversation mode ON — speak freely. Ruchi will keep listening after each reply. Tap ⏹ to stop.');
  await startListening();
}

function endConversation() {
  state.conversationMode = false;
  state.paused = false;
  updateMicUi();
  resetMicUi();
}

function toggleConversation() {
  if (state.conversationMode) {
    pushBubble('system', 'Conversation ended.');
    endConversation();
  } else {
    startConversation();
  }
}

// ---------- Transcript viewer ------------------------------------------
async function loadTranscript() {
  const pre = $('#transcript-text');
  const meta = $('#transcript-meta');
  if (!pre) return;
  try {
    const r = await fetch(`${API_BASE}/api/voice-agent/transcript`);
    const text = await r.text();
    pre.textContent = text || '(empty)';
    pre.scrollTop = pre.scrollHeight;
    try {
      const fr = await fetch(`${API_BASE}/api/voice-agent/transcript/files`);
      const fj = await fr.json();
      const files = (fj.files || []).map((f) => `${f.name} (${f.bytes}B)`).join(', ');
      meta.textContent = files ? `saved: ${files}` : 'no files yet';
    } catch { meta.textContent = ''; }
  } catch (e) {
    pre.textContent = 'Could not load transcript: ' + e.message;
  }
}

function updateVoicePill() {
  const p = $('#voice-pill');
  const serverStt = state.caps?.sarvam || (state.caps?.vosk && state.caps?.vosk_model && state.caps?.ffmpeg);
  const serverTts = state.caps?.elevenlabs || state.caps?.edge_tts;
  const browserStt = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  const browserTts = !!window.speechSynthesis;
  const inOk = serverStt || browserStt;
  const outOk = serverTts || browserTts;
  if (state.conversationMode) {
    p.textContent = '● conversation';
    p.classList.remove('dead'); p.classList.add('live');
  } else if (inOk && outOk) {
    p.textContent = serverStt && serverTts ? 'voice ✓ (server)' : 'voice ✓';
    p.classList.remove('dead'); p.classList.add('live');
  } else {
    p.textContent = `voice ${inOk ? 'in✓' : 'in✗'} ${outOk ? 'out✓' : 'out✗'}`;
    p.classList.remove('live'); p.classList.add('dead');
  }
}

// ---------- Session lifecycle -------------------------------------------
async function createSession(mode) {
  const body = { mode };
  if (mode === 'cook') body.recipe = RECIPE;
  const r = await fetch(`${API_BASE}/api/voice-agent/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`createSession ${r.status}`);
  const j = await r.json();
  state.sessionId = j.session_id;
  state.eventCount = 0;
  state.lastState = j;
  $('#session-id').textContent = j.session_id.slice(0, 8) + '…';
  $('#recipe-name').textContent = j.recipe?.recipeName || RECIPE.recipeName;
  renderRecipeSteps(j.recipe);
  renderIntakeFields(j);
  refreshState(j);
  pushBubble('system', `Session ready — ${mode} mode. ${j.recipe ? j.recipe.recipeName + ' loaded.' : ''}`);
  startPolling();
}

/** Swap the recipe mid-session (user named a new dish). */
async function setRecipe(recipe) {
  RECIPE = recipe;
  $('#recipe-name').textContent = recipe.recipeName;
  renderRecipeSteps({ recipeName: recipe.recipeName, steps: recipe.steps, stepIndex: 0 });
  $('#state-step').textContent = `1 / ${recipe.steps.length}`;
  flashTile('#state-step');
  pushBubble('system', `📖 Switched to ${recipe.recipeName} (${recipe.steps.length} steps).`);
  if (!state.sessionId) return;
  try {
    const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}/recipe`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipe: { ...recipe, stepIndex: 0 } }),
    });
    if (r.ok) {
      const j = await r.json();
      state.lastState = j;
    }
  } catch (e) {
    log('error', 'set recipe failed', String(e).slice(0, 100));
  }
}

async function resetSession() {
  if (state.pollHandle) clearInterval(state.pollHandle);
  if (state.sessionId) {
    try { await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`, { method: 'DELETE' }); } catch {}
  }
  state.sessionId = null;
  $('#transcript').innerHTML = '';
  $('#log').innerHTML = '';
  pushBubble('system', 'New session starting…');
  await createSession(state.mode);
}

// ---------- State (event-driven, instant) ------------------------------
let _timerHandle = null;

function refreshState(j) {
  state.lastState = j;
  $('#state-field').textContent = j.current_field || '—';
  $('#state-step').textContent =
    j.recipe ? `${j.recipe.stepIndex + 1} / ${j.recipe.steps.length}` : '—';
  $('#state-paused').textContent = j.paused ? 'yes' : 'no';
  renderRecipeSteps(j.recipe);
  renderIntakeFields(j);
  renderPausedVisual(j.paused);
  renderTimerTile(j.active_timer);
}

/** Apply only the *new* events since the last turn, instantly. */
function applyEvents(events) {
  if (!Array.isArray(events)) return;
  const fresh = events.slice(state.eventCount || 0);
  state.eventCount = events.length;
  fresh.forEach((ev) => applyEvent(ev));
  if (fresh.length) {
    // keep the session state in sync after applying
    refreshStateFromLocal();
  }
}

function applyEvent(ev) {
  const t = ev.type;
  switch (t) {
    case 'step_changed':
      if (state.lastState?.recipe) {
        state.lastState.recipe.stepIndex = ev.stepIndex;
        renderRecipeSteps(state.lastState.recipe);
      }
      $('#state-step').textContent = state.lastState?.recipe
        ? `${ev.stepIndex + 1} / ${state.lastState.recipe.steps.length}` : '—';
      flashTile('#state-step');
      break;
    case 'recipe_changed':
      state.lastState = state.lastState || {};
      state.lastState.recipe = {
        recipeName: ev.recipeName, steps: ev.steps, stepIndex: ev.stepIndex || 0,
      };
      RECIPE = { recipeName: ev.recipeName, steps: ev.steps };
      $('#recipe-name').textContent = ev.recipeName;
      renderRecipeSteps(state.lastState.recipe);
      flashTile('#state-step');
      break;
    case 'timer_started':
      startTimerCountdown(ev.seconds, ev.label);
      flashTile('#state-timer');
      break;
    case 'timer_cancelled':
      stopTimerCountdown();
      break;
    case 'timer_expired':
      onTimerExpired(ev.label);
      break;
    case 'paused':
      state.paused = true;
      renderPausedVisual(true);
      break;
    case 'resumed':
      state.paused = false;
      renderPausedVisual(false);
      break;
    case 'issue_reported':
      state.lastState = state.lastState || {};
      state.lastState.active_issue = { issue: ev.issue, details: ev.details };
      flashTile('#state-field');
      break;
    case 'issue_resolved':
      if (state.lastState) state.lastState.active_issue = null;
      break;
    case 'field_saved':
      if (state.lastState) {
        state.lastState.answers = state.lastState.answers || {};
        state.lastState.answers[ev.field] = ev.value;
        state.lastState.current_field = ev.next_field || '';
        renderIntakeFields(state.lastState);
        $('#state-field').textContent = ev.next_field || '—';
        flashTile('#state-field');
      }
      break;
    case 'interview_complete':
      pushBubble('system', '✅ Recipe intake complete — all fields saved.');
      break;
    case 'cooking_finished':
      pushBubble('system', '🍽 Cooking finished.');
      stopTimerCountdown();
      break;
    default:
      break;
  }
}

function refreshStateFromLocal() {
  const j = state.lastState;
  if (!j) return;
  $('#state-field').textContent = j.current_field || '—';
  if (j.recipe) {
    $('#state-step').textContent = `${j.recipe.stepIndex + 1} / ${j.recipe.steps.length}`;
    renderRecipeSteps(j.recipe);
  }
  renderIntakeFields(j);
}

function flashTile(sel) {
  const el = $(sel)?.closest('.state-tile');
  if (!el) return;
  el.classList.remove('flash');
  void el.offsetWidth;   // restart animation
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 900);
}

function renderPausedVisual(paused) {
  $('#state-paused').textContent = paused ? 'yes' : 'no';
  const tile = $('#state-paused')?.closest('.state-tile');
  tile?.classList.toggle('alert', !!paused);
  $('#orb').classList.toggle('paused', !!paused);
}

// ---------- Live timer countdown ---------------------------------------
function renderTimerTile(t) {
  if (!t) { $('#state-timer').textContent = '—'; return; }
  // if a countdown is already running, don't clobber it
  if (state.timerEndsAt) return;
  $('#state-timer').textContent = `${t.seconds}s · ${t.label}`;
}

function startTimerCountdown(seconds, label) {
  stopTimerCountdown();
  state.timerEndsAt = Date.now() + seconds * 1000;
  state.timerLabel = label;
  const tick = () => {
    const remain = Math.max(0, Math.round((state.timerEndsAt - Date.now()) / 1000));
    const mm = String(Math.floor(remain / 60)).padStart(2, '0');
    const ss = String(remain % 60).padStart(2, '0');
    $('#state-timer').textContent = `${mm}:${ss} · ${label}`;
    if (remain <= 0) { stopTimerCountdown(); onTimerExpired(label); return; }
    state.timerHandle = setTimeout(tick, 250);
  };
  tick();
}

function stopTimerCountdown() {
  if (state.timerHandle) { clearTimeout(state.timerHandle); state.timerHandle = null; }
  state.timerEndsAt = null;
  state.timerLabel = null;
  $('#state-timer').textContent = '—';
}

function onTimerExpired(label) {
  stopTimerCountdown();
  pushBubble('system', `⏰ Time's up — check ${label}.`);
  const tile = $('#state-timer')?.closest('.state-tile');
  if (tile) {
    tile.classList.add('alert');
    setTimeout(() => tile.classList.remove('alert'), 3000);
  }
  // chime if possible
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const o = ac.createOscillator(); const g = ac.createGain();
    o.connect(g); g.connect(ac.destination);
    o.frequency.value = 880; g.gain.value = 0.15;
    o.start(); o.stop(ac.currentTime + 0.5);
  } catch {}
}

// Slow background poll as a safety net only (events give instant updates).
function startPolling() {
  if (state.pollHandle) clearInterval(state.pollHandle);
  state.pollHandle = setInterval(async () => {
    if (!state.sessionId) return;
    try {
      const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`);
      if (!r.ok) return;
      const j = await r.json();
      refreshState(j);
      state.eventCount = (j.events || []).length;
    } catch {}
  }, 5000);
}

// ---------- Recipe steps render -----------------------------------------
function renderRecipeSteps(recipe) {
  const ol = $('#recipe-steps');
  ol.innerHTML = '';
  if (!recipe) return;
  recipe.steps.forEach((s, i) => {
    const li = document.createElement('li');
    if (i < recipe.stepIndex) li.className = 'done';
    else if (i === recipe.stepIndex) li.className = 'active';
    const num = document.createElement('span');
    num.className = 'num';
    num.textContent = i + 1;
    li.appendChild(num);
    li.appendChild(document.createTextNode(s));
    ol.appendChild(li);
  });
}

// ---------- Intake fields render ----------------------------------------
function renderIntakeFields(j) {
  const card = $('#intake-card');
  const list = $('#intake-fields');
  if (state.mode !== 'intake') { card.style.display = 'none'; return; }
  card.style.display = '';
  list.innerHTML = '';
  INTAKE_FIELDS.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'intake-field';
    const val = j.answers?.[f.name];
    const isCurrent = !val && j.current_field === f.name;
    const isDone = !!val;
    if (isDone) row.classList.add('done');
    else if (isCurrent) row.classList.add('current');
    row.innerHTML = `
      <span class="check">${isDone ? '✓' : (isCurrent ? '…' : '')}</span>
      <span class="name">${f.name}</span>
      <span class="val">${val || ''}</span>
    `;
    list.appendChild(row);
  });
}

// ---------- Conversation / log ------------------------------------------
function pushBubble(kind, text) {
  const t = $('#transcript');
  // Remove empty state if present
  const empty = t.querySelector('.transcript-empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = `bubble ${kind}`;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = kind === 'user' ? 'You' : kind === 'ai' ? 'Ruchi' : '·';
  const body = document.createElement('div');
  body.className = 'text';
  body.textContent = text;
  div.append(who, body);
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
  return div;
}

function pushToolAnnotation(toolName, ok = true) {
  const t = $('#transcript');
  const empty = t.querySelector('.transcript-empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = `tool-annotation ${ok ? '' : 'error'}`;
  div.innerHTML = `<span class="dot"></span> tool fired: <b>${toolName}</b>`;
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
}

function log(kind, name, payload) {
  const entry = document.createElement('div');
  entry.className = `log-entry ${kind}`;
  const head = document.createElement('div');
  head.className = 'head';
  const nameEl = document.createElement('span');
  nameEl.className = 'name';
  nameEl.textContent = name;
  const timeEl = document.createElement('span');
  timeEl.className = 'time';
  timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  head.append(nameEl, timeEl);
  entry.append(head);
  if (payload !== undefined) {
    const pre = document.createElement('pre');
    pre.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
    entry.append(pre);
  }
  const logEl = $('#log');
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

// ---------- Orb state ----------------------------------------------------
function setOrb(stateName, label) {
  state.orbState = stateName;
  const orb = $('#orb');
  const status = $('#orb-status');
  orb.classList.remove('idle', 'listening', 'thinking', 'speaking', 'tool');
  if (stateName !== 'idle') orb.classList.add(stateName);
  else orb.classList.add('idle');
  status.classList.remove('listening', 'thinking', 'speaking', 'tool');
  if (stateName !== 'idle') status.classList.add(stateName);
  status.textContent = label || stateName;
}

// ---------- Message send ------------------------------------------------
async function sendMessage(text) {
  if (!text.trim() || state.busy) return;
  state.busy = true;
  $('#input-text').value = '';
  pushBubble('user', text);
  setOrb('thinking', 'thinking…');

  // If the user is *requesting* a dish, swap the recipe panel + session
  // instantly. We require an intent word so "the biryani is salty" doesn't
  // switch recipes mid-cook.
  if (state.mode === 'cook') {
    const dish = detectDish(text);
    if (dish && isDishRequest(text) && dish.recipeName !== RECIPE.recipeName) {
      await setRecipe(dish);
    }
  }

  try {
    const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}/turn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const j = await r.json();

    // Apply state changes from events immediately (no polling lag).
    applyEvents(j.events);

    // Annotate every tool that fired
    (j.tool_calls || []).forEach((tc, i) => {
      const tr = (j.tool_results || [])[i] || {};
      const ok = !tr.error;
      pushToolAnnotation(tc.name, ok);
      log(ok ? 'success' : 'error', `${ok ? '←' : '✗'} ${tc.name}`, {
        args: tc.args,
        result: tr.result,
        error: tr.error,
      });
      // The `pause` tool means "stop talking to me" — stop the loop.
      if (tc.name === 'pause') state.paused = true;
      // `resume` clears the pause and lets the loop continue.
      if (tc.name === 'resume') state.paused = false;
      // `finish_cooking` ends the session.
      if (tc.name === 'finish_cooking') state.finishRequested = true;
    });

    pushBubble('ai', j.assistant_text);
    await speak(j.assistant_text);   // wait until Ruchi finishes speaking
    refreshState(await getState());
  } catch (err) {
    pushBubble('system', `⚠ ${err.message}`);
    log('error', 'request failed', err.message);
  } finally {
    state.busy = false;
    // If the model asked to pause or finish, stop the loop.
    if (state.finishRequested) {
      state.finishRequested = false;
      pushBubble('system', 'Cooking session finished.');
      endConversation();
      return;
    }
    if (state.paused) {
      pushBubble('system', 'Paused — tap the orb to resume the conversation.');
      endConversation();
      return;
    }
    // Otherwise continue the hands-free loop.
    await afterTurn();
  }
}

async function getState() {
  const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`);
  return r.json();
}

// ---------- Direct tool invocation (chips) -----------------------------
async function callToolDirect(toolName, args, btn) {
  if (state.busy) return;
  state.busy = true;

  // Instant visual feedback on the chip itself.
  if (btn) {
    btn.classList.add('firing');
    setTimeout(() => btn.classList.remove('firing'), 700);
  }
  setOrb('tool', `→ ${toolName}`);
  log('system', `→ ${toolName}`, args);

  try {
    // Every tool goes through the direct-call endpoint so pressing
    // "start_timer" fires exactly start_timer — no LLM in the loop.
    const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: toolName, args }),
    });
    const j = await r.json();

    if (!r.ok || j.detail) {
      pushToolAnnotation(toolName, false);
      log('error', `✗ ${toolName}`, j.detail || j);
      return;
    }

    pushToolAnnotation(toolName, true);
    log('success', `← ${toolName}`, j.result);

    // Apply the resulting events instantly (step change, timer, pause…).
    applyEvents(j.events);

    // Speak the canned phrase if the handler produced one.
    const phrase = j.result?.phrase || j.result?.lead_in;
    if (phrase) {
      pushBubble('ai', phrase);
      await speak(phrase);
    }

    // Keep the full session snapshot in sync.
    if (j.session) refreshState(j.session);
  } catch (err) {
    pushToolAnnotation(toolName, false);
    log('error', `← ${toolName}`, err.message);
  } finally {
    state.busy = false;
    setTimeout(() => {
      if (state.orbState === 'tool') setOrb('idle', 'tap to talk');
    }, 600);
  }
}

function synthesize(toolName, args) {
  switch (toolName) {
    case 'finish_interview': return 'finish';
    case 'next_step': return 'okay next';
    case 'previous_step': return 'go back';
    case 'goto_step': return `go to step ${args.step}`;
    case 'start_timer': return `set a timer for ${args.seconds} seconds`;
    case 'cancel_timer': return 'cancel the timer';
    case 'pause': return 'pause.';
    case 'resume': return 'continue';
    case 'report_issue': return `this is too ${args.issue}`;
    case 'resolve_issue': return 'fixed, thanks';
    case 'finish_cooking': return 'finish cooking';
    default: return toolName;
  }
}

function gatherArgs(tool) {
  switch (tool) {
    case 'save_answer':
      return {
        field: $('#intake-field').value.trim(),
        value: $('#intake-value').value.trim(),
      };
    case 'goto_step': {
      const n = parseInt($('#goto-step').value, 10);
      if (!Number.isFinite(n) || n < 1) {
        log('error', 'bad arg', { tool, reason: 'step must be ≥ 1' });
        return null;
      }
      return { step: n };
    }
    case 'start_timer': {
      const s = parseInt($('#timer-seconds').value, 10);
      if (!Number.isFinite(s) || s < 1) {
        log('error', 'bad arg', { tool, reason: 'seconds must be ≥ 1' });
        return null;
      }
      return { seconds: s, label: $('#timer-label').value.trim() || 'your timer' };
    }
    case 'report_issue':
      return { issue: $('#issue-cat').value, details: $('#issue-details').value.trim() };
    default:
      return {};
  }
}

// ---------- Mode switching ----------------------------------------------
async function switchMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  $('#mode-intake').classList.toggle('active', mode === 'intake');
  $('#mode-cook').classList.toggle('active', mode === 'cook');
  $('#group-intake').style.display = mode === 'intake' ? '' : 'none';
  if (state.sessionId) {
    try { await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`, { method: 'DELETE' }); } catch {}
  }
  state.sessionId = null;
  $('#transcript').innerHTML = '';
  await createSession(mode);
  // Speak the first intake question
  if (mode === 'intake') {
    const q = INTAKE_FIELDS[0].question;
    pushBubble('ai', q);
    speak(q);
  } else {
    const greeting = "నమస్కారం! నేను రుచి. Let's cook " + RECIPE.recipeName + '. ' + RECIPE.steps[0];
    pushBubble('ai', greeting);
    speak(greeting);
  }
}

// ---------- UI wiring ----------------------------------------------------
function wireUi() {
  $('#mode-intake').addEventListener('click', () => switchMode('intake'));
  $('#mode-cook').addEventListener('click', () => switchMode('cook'));

  $('#input-form').addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage($('#input-text').value);
  });

  // Mic button toggles hands-free conversation mode on/off.
  $('#mic-btn').addEventListener('click', () => {
    toggleConversation();
  });

  // Orb does the same.
  $('#orb').addEventListener('click', () => {
    toggleConversation();
  });

  $('#reset-btn').addEventListener('click', resetSession);

  // Transcript panel
  $('#transcript-refresh')?.addEventListener('click', loadTranscript);
  $('#transcript-clear')?.addEventListener('click', async () => {
    if (!confirm('Clear all transcripts on the server?')) return;
    try {
      await fetch(`${API_BASE}/api/voice-agent/transcript`, { method: 'DELETE' });
      await loadTranscript();
    } catch (e) { log('error', 'clear transcript failed', String(e)); }
  });

  $$('.side-tabs button').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.side-tabs button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      $$('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === target));
      if (target === 'transcript') loadTranscript();
    });
  });

  $$('.chip[data-tool]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const tool = btn.dataset.tool;
      const args = gatherArgs(tool);
      if (args === null) return;
      await callToolDirect(tool, args, btn);
    });
  });

  // Keyboard: enter to send
  $('#input-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage($('#input-text').value);
    }
  });
}

// ---------- Boot --------------------------------------------------------
(async function boot() {
  await loadVoices();
  await loadCaps();
  wireUi();
  await ping();
  setInterval(ping, 5000);
  updateVoicePill();
  await createSession('cook');

  // Diagnostics: tell the user exactly which providers are active.
  const c = state.caps || {};
  const inMode = c.sarvam ? `Sarvam ${c.sarvam_model || 'saaras:v3'} (cloud)` :
                 ((c.vosk && c.vosk_model && c.ffmpeg) ? 'Vosk (offline Telugu)' :
                  ((window.SpeechRecognition || window.webkitSpeechRecognition) ? 'browser Web Speech' : 'none'));
  const outMode = c.elevenlabs ? `ElevenLabs ${c.elevenlabs_model || 'eleven_v3_conversational'}` :
                  (c.edge_tts ? 'edge-tts (Telugu neural)' :
                   (window.speechSynthesis ? 'browser speechSynthesis' : 'none'));
  const llmMode = c.gemini ? `Gemini ${c.gemini_model || ''}` : 'mock runner (no GOOGLE_API_KEY)';
  pushBubble('system', `Voice in: ${inMode} · Voice out: ${outMode} · LLM: ${llmMode}`);
  if (inMode === 'none') {
    pushBubble('system', '⚠ No voice input available — type your messages.');
  }
  if (outMode === 'none') {
    pushBubble('system', '⚠ No voice output available.');
  }

  // Greet on first load (spoken via edge-tts if available).
  const greeting = "నమస్కారం! నేను రుచి. Let's cook " + RECIPE.recipeName + '. Step 1: ' + RECIPE.steps[0];
  pushBubble('ai', greeting);
  speak(greeting);
})();
