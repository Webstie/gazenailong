"""Cross-platform audio + text-to-speech.

Picks the best backend available on macOS, Windows, or Linux:

  TTS:    pyttsx3 (mac/win/linux)  ->  espeak-ng pipe (linux)  ->  silent
  Alerts: afplay (mac) / winsound (win) / aplay (linux) / simpleaudio

The original Orange Pi build hard-coded espeak-ng + ALSA `aplay`; this
generalises that so the same code runs on a laptop.
"""

import os
import sys
import time
import queue
import threading
import subprocess

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

if IS_WIN:
    try:
        import winsound
    except Exception:
        winsound = None
else:
    winsound = None

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

        if self.tts_backend == "say":
            self.voice_name = self._pick_say_voice()
        elif self.tts_backend == "pyttsx3":
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 170)
                self._engine.setProperty("volume", 1.0)
                self.voice_name = self._select_female_voice(self._engine)
            except Exception:
                self.tts_backend = "none"
                self.enabled = pyttsx3 is None  # nothing usable

        threading.Thread(target=self._speech_worker, daemon=True).start()

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

        Prefers Premium/Enhanced English female voices (these sound far more
        natural than the default Samantha). Override with GAZE_VOICE_NAME.
        """
        pref = os.environ.get("GAZE_VOICE_NAME")
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
        female_kw = ("samantha", "ava", "zoe", "allison", "susan", "victoria",
                     "karen", "moira", "tessa", "fiona", "serena", "nicky",
                     "joelle", "kathy", "princess")
        pool = [(n, l) for n, l in english
                if any(f in n.lower() for f in female_kw)] or english

        def score(name):
            nl = name.lower()
            s = 0
            if "premium" in nl:
                s += 4
            elif "enhanced" in nl:
                s += 3
            if "samantha" in nl:
                s += 1
            return s

        pool.sort(key=lambda x: score(x[0]), reverse=True)
        return pool[0][0] if pool else pref

    # ---------- speech ----------
    def _synth(self, text):
        if self.tts_backend == "say":
            args = ["say"]
            if self.voice_name:
                args += ["-v", self.voice_name]
            args += ["-r", "180", str(text)]
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
