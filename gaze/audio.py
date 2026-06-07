"""Cross-platform audio + text-to-speech.

Picks the best backend available on macOS, Windows, or Linux. On macOS we
default to **Microsoft Edge TTS** — its neural "Multilingual" voices (Ava,
Andrew, Emma, Brian) are the highest-quality free voices available and are
what Microsoft Copilot Voice uses. They require internet; on offline /
endpoint-failure we transparently fall back to the macOS `say` command.

  TTS:    edge-tts (cloud neural, free)  ->  say (mac native)  ->
          pyttsx3 (mac/win/linux)        ->  espeak-ng (linux)  ->  silent
  Alerts: afplay (mac) / winsound (win) / aplay (linux)

The voice ID format encodes which backend handles it:
  * "en-XX-NameNeural"  → Edge TTS (looks like a Microsoft short name)
  * everything else     → macOS `say` voice name
"""

import logging
import os
import re
import sys
import tempfile
import time
import queue
import threading
import subprocess

log = logging.getLogger("gaze")

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# edge-tts pulls in aiohttp; both are pure Python wheels and bundle fine.
try:
    import asyncio
    import edge_tts            # noqa: F401  -- imported lazily inside calls
    _HAS_EDGE_TTS = True
except Exception:
    _HAS_EDGE_TTS = False

if IS_WIN:
    try:
        import winsound
    except Exception:
        winsound = None
else:
    winsound = None

# Curated subset of Microsoft Edge / Azure Neural voices, in display order.
# The "Multilingual" variants are Microsoft's newest (used by Copilot Voice)
# and are markedly more natural than the older single-language Neural ones.
_EDGE_VOICES = [
    ("en-US-AvaMultilingualNeural",
     "Ava — calm, conversational (US)", "en_US"),
    ("en-US-AndrewMultilingualNeural",
     "Andrew — warm, conversational (US)", "en_US"),
    ("en-US-EmmaMultilingualNeural",
     "Emma — friendly, upbeat (US)", "en_US"),
    ("en-US-BrianMultilingualNeural",
     "Brian — relaxed, friendly (US)", "en_US"),
    ("en-US-AvaNeural",         "Ava — older Neural (US)", "en_US"),
    ("en-US-AndrewNeural",      "Andrew — older Neural (US)", "en_US"),
    ("en-US-AriaNeural",        "Aria — clear, news-anchor (US)", "en_US"),
    ("en-US-JennyNeural",       "Jenny — friendly assistant (US)", "en_US"),
    ("en-US-GuyNeural",         "Guy — friendly assistant (US)", "en_US"),
    ("en-US-ChristopherNeural", "Christopher — calm narrator (US)", "en_US"),
    ("en-GB-SoniaNeural",       "Sonia — British (UK)", "en_GB"),
    ("en-GB-RyanNeural",        "Ryan — British (UK)", "en_GB"),
    ("en-AU-NatashaNeural",     "Natasha — Australian (AU)", "en_AU"),
]

_DEFAULT_EDGE_VOICE = "en-US-AvaMultilingualNeural"

# Microsoft Edge short-name pattern (e.g. en-US-AvaNeural). Used to route
# a stored voice id to the right backend without an explicit prefix.
_EDGE_ID_RE = re.compile(r"^[a-z]{2}-[A-Z]{2}-")


def _is_edge_voice(voice_id):
    return bool(voice_id and _EDGE_ID_RE.match(voice_id))

# Optional ALSA device override for Linux/Orange Pi parity (env-configurable).
AUDIO_DEVICE_LINUX = os.environ.get("GAZE_ALSA_DEVICE", "default")
ESPEAK_BIN = "espeak-ng"
# "+f3" selects a female voice variant in espeak-ng
ESPEAK_VOICE, ESPEAK_SPEED, ESPEAK_PITCH, ESPEAK_GAP = "en-us+f3", 145, 45, 5


def _which(name):
    from shutil import which
    return which(name) is not None


class AudioManager:
    """Threaded speech queue + a looping alert tone, output-device aware.

    Speech and the warning loop run on their own threads so the detection
    loop never blocks. `speak(..., priority=True)` interrupts whatever is
    playing, matching the original behaviour.
    """

    def __init__(self, alert_path=None):
        self._q = queue.Queue()
        self._last = {}
        self._stop = threading.Event()
        self._engine = None
        self._proc = None
        self._proc_lock = threading.Lock()
        self.voice_name = None

        # alert-chime state
        self._alert_path = alert_path
        self._alert_proc = None
        self._alert_lock = threading.Lock()
        self._last_alert = 0.0

        self.tts_backend = self._detect_tts()
        self.enabled = self.tts_backend != "none"

        # voice_name holds the ID we route on. For Edge TTS it's a Microsoft
        # short name (e.g. "en-US-AvaMultilingualNeural"); for macOS `say`
        # it's the say voice name (e.g. "Samantha", "Flo (English (US))").
        self.voice_name = self._load_persisted_voice()

        if self.tts_backend == "say":
            # Initialise pyttsx3 lazily only if needed — on macOS with
            # Edge TTS available, say handles the fallback path without it.
            if self.voice_name is None:
                if _HAS_EDGE_TTS and IS_MAC:
                    self.voice_name = _DEFAULT_EDGE_VOICE
                else:
                    self.voice_name = self._pick_say_voice()
        elif self.tts_backend == "pyttsx3":
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 150)
                self._engine.setProperty("volume", 1.0)
                if self.voice_name is None:
                    self.voice_name = self._select_female_voice(self._engine)
            except Exception:
                self.tts_backend = "none"
                self.enabled = pyttsx3 is None  # nothing usable

        threading.Thread(target=self._speech_worker, daemon=True).start()

    @staticmethod
    def _load_persisted_voice():
        """Return the voice ID the user pinned via the sidebar picker, or
        None if they haven't picked one yet. Stored at ~/.gazenailong/voice."""
        pref = os.environ.get("GAZE_VOICE_NAME")
        if pref:
            return pref
        try:
            cfg = os.path.join(os.path.expanduser("~"),
                               ".gazenailong", "voice")
            with open(cfg) as f:
                v = f.read().strip()
                return v or None
        except (FileNotFoundError, OSError):
            return None

    # ---------- voice selection ----------
    # Known female voices across macOS / Windows / Linux espeak.
    _FEMALE_NAMES = [
        "samantha", "victoria", "allison", "ava", "susan", "zoe", "serena",
        "karen", "moira", "tessa", "fiona", "kate", "zira", "hazel", "female",
    ]

    @staticmethod
    def _is_english(v):
        """True if a pyttsx3 voice looks like an English voice."""
        hay = (getattr(v, "id", "") + " " + getattr(v, "name", "")).lower()
        langs = getattr(v, "languages", None) or []
        langstr = " ".join(str(x).lower() for x in langs)
        for tag in ("en-us", "en_us", "en-gb", "en_gb", "english", "en-au"):
            if tag in hay or tag in langstr:
                return True
        # bare "en" only from the languages list (avoid matching words like "open")
        return any(s.lstrip("b'").startswith("en") for s in langstr.split())

    @classmethod
    def _select_female_voice(cls, engine):
        """Pick an ENGLISH female voice; return its name (or None for default).

        Order matters: we try well-known good English female voices by name
        first (e.g. Samantha), so we never accidentally grab a female voice in
        the wrong language (Alice is Italian, which is what made it sound bad).
        """
        try:
            voices = engine.getProperty("voices")
        except Exception:
            return None

        # 1) preferred English female voices, by name, in priority order
        for target in cls._FEMALE_NAMES:
            for v in voices:
                hay = (getattr(v, "name", "") + " " + getattr(v, "id", "")).lower()
                if target in hay and cls._is_english(v):
                    engine.setProperty("voice", v.id)
                    return v.name
        # 2) any English voice flagged female by the driver
        for v in voices:
            g = getattr(v, "gender", None)
            if g and "female" in str(g).lower() and cls._is_english(v):
                engine.setProperty("voice", v.id)
                return v.name
        # 3) give up gracefully — keep the system default (still English)
        return None

    # ---------- backend detection ----------
    @staticmethod
    def _detect_tts():
        # On macOS the native `say` command is the best option: it can use the
        # high-quality Premium/Enhanced voices and is cleanly interruptible.
        if IS_MAC and _which("say"):
            return "say"
        if pyttsx3 is not None:
            return "pyttsx3"
        if IS_LINUX and _which(ESPEAK_BIN):
            return "espeak"
        return "none"

    @staticmethod
    def _pick_say_voice():
        """Choose the best installed macOS voice via `say -v '?'`.

        Prefers Premium/Enhanced English female voices, then the macOS Sonoma+
        neural personality voices (Flo, Shelley, Sandy, ...), then falls back
        to standard Samantha. Override at runtime with either the
        GAZE_VOICE_NAME env var, or by writing the voice name to
        ~/.gazenailong/voice (works in the packaged .app where env vars from
        the user's shell are not inherited).
        """
        pref = os.environ.get("GAZE_VOICE_NAME")
        if not pref:
            try:
                cfg = os.path.join(os.path.expanduser("~"),
                                   ".gazenailong", "voice")
                with open(cfg) as f:
                    pref = f.read().strip() or None
            except (FileNotFoundError, OSError):
                pass
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, timeout=5).stdout
        except Exception:
            return pref
        voices = []
        for line in out.splitlines():
            head = line.split("#", 1)[0].rstrip()
            parts = head.split()
            if len(parts) < 2:
                continue
            locale, name = parts[-1], " ".join(parts[:-1])
            voices.append((name, locale))
        if pref:
            for n, _ in voices:
                if pref.lower() in n.lower():
                    return n
        english = [(n, l) for n, l in voices if l.lower().startswith("en")]
        # Two families of "natural-sounding" voices on macOS:
        #   * Classic concatenative female voices (Samantha, Ava, Zoe ...) —
        #     ideally with a (Premium) or (Enhanced) suffix.
        #   * macOS Sonoma+ neural personality voices (Flo, Shelley, Sandy,
        #     Reed, Rocko, Eddy ...) — these ship with every modern macOS
        #     install and sound markedly better than the bare 2009-era
        #     Samantha that comes with a fresh system.
        pleasant_kw = (
            "samantha", "ava", "zoe", "allison", "susan", "victoria",
            "karen", "moira", "tessa", "fiona", "serena", "nicky",
            "joelle", "kathy", "princess",
            "flo", "shelley", "sandy", "reed", "rocko", "eddy",
            "grandma", "grandpa", "junior",
        )
        pool = [(n, l) for n, l in english
                if any(f in n.lower() for f in pleasant_kw)] or english

        def score(name):
            nl = name.lower()
            s = 0
            # Quality tier dominates everything else. A Premium / Enhanced
            # variant of *any* voice beats a standard variant of the best one.
            if "premium" in nl:
                s += 20
            elif "enhanced" in nl:
                s += 10
            # Within a tier, prefer the warmest classic voices when available.
            if "ava" in nl:
                s += 5
            elif "zoe" in nl:
                s += 4
            elif "allison" in nl:
                s += 3
            elif "serena" in nl:
                s += 2
            # Sonoma+ neural picks. Flo is the warmest, most "friend-like"
            # of the new personality voices; Shelley and Sandy are close.
            # All of them beat the *standard* Samantha (no suffix), which is
            # the brittle 2009 voice nobody likes.
            if "flo" in nl:
                s += 4
            elif "shelley" in nl:
                s += 3
            elif "sandy" in nl:
                s += 3
            elif "reed" in nl:
                s += 2
            elif "rocko" in nl:
                s += 2
            elif "eddy" in nl:
                s += 2
            elif "samantha" in nl:
                # only wins if it's actually (Premium) or (Enhanced); bare
                # Samantha falls to the bottom of the modern-voice pile.
                s += 1
            return s

        # Prefer en_US over en_GB / en_AU when scores tie — that's what the
        # bare `say` default is and matches the previous Samantha behaviour.
        def locale_bonus(locale):
            l = locale.lower()
            if l in ("en_us", "en-us"):
                return 2
            if l in ("en_gb", "en-gb"):
                return 1
            return 0

        pool.sort(key=lambda x: (score(x[0]), locale_bonus(x[1])), reverse=True)
        return pool[0][0] if pool else pref

    # ---------- runtime voice swap (sidebar picker) ----------
    @staticmethod
    def list_english_voices():
        """Enumerate installed English voices with a quality tier.

        Returns a list of `{"id", "name", "locale", "quality"}` dicts. The
        `id` is what gets passed to `apply_voice` / `preview_voice` and is
        what we route on; `name` is what the picker UI displays.

        Order: AI (Edge TTS, free neural cloud) → Premium → Enhanced →
        macOS Sonoma+ Neural → Standard. en_US wins ties over en_GB / en_AU.
        """
        # Cloud neural voices first, in our hand-curated quality order so
        # the Multilingual variants (Microsoft's newest, used by Copilot)
        # appear above the older single-language Neural ones.
        edge_voices = []
        if _HAS_EDGE_TTS:
            for vid, label, locale in _EDGE_VOICES:
                edge_voices.append({"id": vid, "name": label,
                                    "locale": locale, "quality": "ai"})

        # macOS native `say` voices.
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, timeout=5).stdout
        except Exception:
            out = ""
        NEURAL = ("flo", "shelley", "sandy", "reed", "rocko", "eddy",
                  "grandma", "grandpa", "junior")
        say_voices = []
        for line in out.splitlines():
            head = line.split("#", 1)[0].rstrip()
            parts = head.split()
            if len(parts) < 2:
                continue
            locale, name = parts[-1], " ".join(parts[:-1])
            if not locale.lower().startswith("en"):
                continue
            nl = name.lower()
            if "premium" in nl:
                q = "premium"
            elif "enhanced" in nl:
                q = "enhanced"
            elif any(k in nl for k in NEURAL):
                q = "neural"
            else:
                q = "standard"
            say_voices.append({"id": name, "name": name,
                               "locale": locale, "quality": q})
        order = {"premium": 0, "enhanced": 1, "neural": 2, "standard": 3}
        say_voices.sort(key=lambda v: (
            order[v["quality"]],
            0 if v["locale"].lower() in ("en_us", "en-us") else 1,
            v["name"].lower(),
        ))

        return edge_voices + say_voices

    @staticmethod
    def preview_voice(voice_id, text=None):
        """Speak a representative sample with the given voice. Non-blocking.

        Dispatches by ID format: Microsoft "en-XX-FooNeural" shorts go to
        Edge TTS, anything else goes to `say`. Edge TTS is async (MP3 over
        HTTPS then `afplay`), so we drive it from a daemon thread.
        """
        if not voice_id:
            return False
        text = text or ("Hey, you've been distracted for a while. "
                        "Want to come back to what you were doing?")
        if _is_edge_voice(voice_id):
            threading.Thread(
                target=AudioManager._edge_speak_blocking,
                args=(str(voice_id), str(text), None),
                daemon=True,
            ).start()
            return True
        try:
            subprocess.Popen(
                ["say", "-v", voice_id, "-r", "155", str(text)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def apply_voice(self, voice_id):
        """Pin a voice as the new default AND persist for next launch."""
        if not voice_id:
            return None
        self.voice_name = voice_id
        try:
            cfg_dir = os.path.join(os.path.expanduser("~"), ".gazenailong")
            os.makedirs(cfg_dir, exist_ok=True)
            with open(os.path.join(cfg_dir, "voice"), "w") as f:
                f.write(voice_id + "\n")
        except Exception:
            log.exception("Failed to persist voice selection")
        return voice_id

    # ---------- Edge TTS plumbing ----------
    @staticmethod
    def _edge_speak_blocking(voice_id, text, proc_holder):
        """Synthesize via edge-tts, save to MP3, play via afplay/aplay.

        `proc_holder` is a callable invoked with the player Popen so the
        main speech-queue loop can kill the current utterance on a
        priority interrupt. Pass None for fire-and-forget previews.

        Returns True on success, False if any step fails (caller falls back
        to `say`).
        """
        if not _HAS_EDGE_TTS:
            return False
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        try:
            # New event loop per call — these are short-lived utterances
            # and we don't want to depend on a shared one.
            import edge_tts as _et
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    _et.Communicate(text, voice_id).save(tmp.name))
            finally:
                loop.close()
            player = "afplay" if IS_MAC else ("aplay" if IS_LINUX else None)
            if player is None:
                return False
            proc = subprocess.Popen([player, tmp.name],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            if proc_holder is not None:
                proc_holder(proc)
            proc.wait()
            return True
        except Exception:
            log.warning("edge-tts synthesis failed for voice=%s",
                        voice_id, exc_info=True)
            return False
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ---------- speech ----------
    def _synth(self, text):
        # Route by the voice ID format. An Edge TTS short-name (e.g.
        # "en-US-AvaMultilingualNeural") goes through edge-tts; anything
        # else uses the platform-native backend.
        if _is_edge_voice(self.voice_name):
            def hold(p):
                with self._proc_lock:
                    self._proc = p

            def release():
                with self._proc_lock:
                    self._proc = None

            ok = self._edge_speak_blocking(self.voice_name, str(text), hold)
            release()
            if ok:
                return
            # Edge failed (no internet, endpoint down, ...). Fall through to
            # the platform-native backend with no specific voice so the user
            # still hears the notification.
            log.info("Falling back to native TTS for this utterance")

        if self.tts_backend == "say":
            args = ["say"]
            # When voice_name is an Edge ID we just failed on, omit -v so
            # `say` picks its system default rather than barking an error.
            if self.voice_name and not _is_edge_voice(self.voice_name):
                args += ["-v", self.voice_name]
            # 155 wpm is noticeably calmer than the macOS default — closer to a
            # supportive friend's pace than a news anchor.
            args += ["-r", "155", str(text)]
            proc = subprocess.Popen(args)
            with self._proc_lock:
                self._proc = proc
            proc.wait()
            with self._proc_lock:
                self._proc = None
            return
        if self.tts_backend == "pyttsx3" and self._engine is not None:
            self._engine.say(text)
            self._engine.runAndWait()
            return
        if self.tts_backend == "espeak":
            safe = str(text).replace('"', '\\"')
            cmd = (f'{ESPEAK_BIN} -v {ESPEAK_VOICE} -s {ESPEAK_SPEED} '
                   f'-p {ESPEAK_PITCH} -g {ESPEAK_GAP} --stdout "{safe}" '
                   f'| aplay -D {AUDIO_DEVICE_LINUX}')
            proc = subprocess.Popen(cmd, shell=True)
            with self._proc_lock:
                self._proc = proc
            proc.wait()
            with self._proc_lock:
                self._proc = None

    def _kill_current(self):
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=0.5)
                except Exception:
                    pass
                self._proc = None
        # pyttsx3 has no clean interrupt mid-utterance; queue clearing handles it.

    def _speech_worker(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._synth(text)
            except Exception:
                pass

    def speak(self, key, text, cooldown=3.0, priority=False):
        if not self.enabled:
            return
        now = time.time()
        if not priority and now - self._last.get(key, 0) < cooldown:
            return
        self._last[key] = now
        if priority:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._kill_current()
        self._q.put(text)

    # ---------- alert chime (single-shot, rate-limited) ----------
    # NOTE: the old build looped warning.wav forever on a worker thread, which
    # is why it "repeated infinitely". Now we play it at most once every
    # ALERT_MIN_INTERVAL seconds while distracted — no loop, no thread.
    def warn_audio_start(self, min_interval=None):
        """Play the alert sound once, rate-limited. Safe to call every frame."""
        if not self._alert_path:
            return
        from . import config as C
        gap = C.ALERT_MIN_INTERVAL if min_interval is None else min_interval
        now = time.time()
        with self._alert_lock:
            if now - self._last_alert < gap:
                return
            self._last_alert = now
            try:
                if IS_MAC:
                    self._alert_proc = subprocess.Popen(
                        ["afplay", self._alert_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif IS_WIN and winsound is not None:
                    winsound.PlaySound(self._alert_path,
                                       winsound.SND_FILENAME | winsound.SND_ASYNC)
                elif IS_LINUX:
                    self._alert_proc = subprocess.Popen(
                        ["aplay", "-D", AUDIO_DEVICE_LINUX, self._alert_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def warn_audio_stop(self):
        """Stop a currently-playing chime (e.g. when the user refocuses)."""
        with self._alert_lock:
            if self._alert_proc and self._alert_proc.poll() is None:
                try:
                    self._alert_proc.terminate()
                except Exception:
                    pass
            self._alert_proc = None
            if IS_WIN and winsound is not None:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        self.warn_audio_stop()
        self._kill_current()
