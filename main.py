import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import subprocess
import threading
import queue
from collections import deque, Counter
from enum import Enum

# Prefer eSpeak-ng piped to aplay on Linux so we can force the ALSA device.
try:
    import pyttsx3  # optional fallback (may be unavailable on Orange Pi)
except Exception:
    pyttsx3 = None

USE_ESPEAK_PIPE = sys.platform.startswith("linux")
ESPEAK_BIN = "espeak-ng"        # change to "espeak" if you only have espeak

# Piper neural TTS (best quality, optional).
# Install: https://github.com/rhasspy/piper
# Set PIPER_BIN to the piper executable path, and PIPER_MODEL to the .onnx model.
# Leave either as "" to skip piper and fall back to espeak-ng.
PIPER_BIN   = "/usr/local/bin/piper"
PIPER_MODEL = "/home/orangepi/piper-voices/en_US-amy-medium.onnx"
PIPER_LIB   = "/home/orangepi/piper"   # folder containing piper's .so libraries

# espeak-ng voice tuning (used when piper is not available)
# -v  voice: en-us, en-gb, mb/mb-en1 (mbrola, much better if installed)
# -s  speed in words/min  (default 175 — lower = more natural)
# -p  pitch 0-99          (default 50  — lower = deeper)
# -g  gap between words   (default 0   — small gap improves clarity)
ESPEAK_VOICE = "en-us"
ESPEAK_SPEED = 145
ESPEAK_PITCH = 40
ESPEAK_GAP   = 5

mp_face_mesh = mp.solutions.face_mesh

DEBUG_VIS = False

# -------- FaceMesh indices --------
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362

IRIS_POINTS = list(range(468, 478))

FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152
NOSE_TIP = 1

# -------- Calibration thresholds --------
CALIB_SECONDS = 5.0         # seconds of *good* frames required
CALIB_MIN_SAMPLES = 30

CALIB_FACE_SIZE_MIN = 0.20  # face_width / frame_width — minimum to resolve features

# -------- Calibration streak --------
CALIB_BAD_STREAK_RESET = 10  # consecutive bad frames before resetting progress

# -------- Detection thresholds --------
EAR_OPEN_THRESH   = 0.20
EYE_REL_THRESH    = 0.12
HEAD_YAW_THRESH   = 0.15
HEAD_PITCH_THRESH = 0.15

# -------- Pose smoothing (EMA) — dead band for minor head movements --------
# Lower alpha = more smoothing (less responsive to quick twitches)
YAW_SMOOTH_ALPHA   = 0.15
PITCH_SMOOTH_ALPHA = 0.15

# -------- Warning state machine --------
FOCUS_WINDOW_SECONDS = 10.0
WARN_ENTER_RATIO = 0.60     # enter warning when bad_ratio >= this
WARN_EXIT_RATIO = 0.30      # exit warning when bad_ratio drops below this (hysteresis)
WARN_ESCALATE_SEC = 15.0    # seconds in warning before escalating to urgent message

NO_FACE_PROMPT_SEC = 2.0

AUDIO_DEVICE_LINUX = "plughw:1,0"
WARNING_AUDIO_CANDIDATES = ["warning.wav", "warning.mp3", "warning.aiff"]

# -------- Focus states --------
class FocusState(Enum):
    FOCUSED = "focused"
    DROWSY = "drowsy"
    LOOKING_AWAY = "looking_away"
    HEAD_TURNED = "head_turned"
    NO_FACE = "no_face"


# -------- warning.wav control --------
warning_active = False
warning_stop_event = threading.Event()
warning_process = None
warning_thread = None
warning_lock = threading.Lock()


# -------- Speech --------
def _detect_tts_backend():
    """
    Pick the best available TTS backend on this system.
    Returns one of: "piper", "espeak_mbrola", "espeak", "pyttsx3", "none"
    """
    if USE_ESPEAK_PIPE:
        # 1. Piper neural TTS — best quality
        if PIPER_BIN and PIPER_MODEL and os.path.exists(PIPER_MODEL) and os.path.exists(PIPER_BIN):
            return "piper"
        # 2. espeak-ng with mbrola voice — only if the mbrola binary actually exists
        mbrola_ok = (
            subprocess.run(["which", "mbrola"], capture_output=True).returncode == 0
            and subprocess.run(["which", "mbrowrap"], capture_output=True).returncode == 0
        )
        if mbrola_ok:
            return "espeak_mbrola"
        # 3. Plain espeak-ng with tuned parameters
        return "espeak"
    # 4. pyttsx3 on non-Linux
    if pyttsx3 is not None:
        return "pyttsx3"
    return "none"


class SpeechManager:
    def __init__(self):
        self._q      = queue.Queue()
        self._last   = {}
        self._stop   = threading.Event()
        self._engine = None
        self._proc   = None          # currently running synthesis subprocess
        self._proc_lock = threading.Lock()

        self.backend = _detect_tts_backend()
        self.enabled = self.backend != "none"

        if self.backend == "pyttsx3":
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 150)
                self._engine.setProperty("volume", 1.0)
                voices = self._engine.getProperty("voices")
                female = next((v for v in voices if "female" in v.name.lower()), None)
                if female:
                    self._engine.setProperty("voice", female.id)
            except Exception:
                self.enabled = False

        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def _kill_current(self):
        """Terminate the currently running synthesis subprocess, if any."""
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=0.5)
                except Exception:
                    pass
                self._proc = None

    def _synth(self, text: str):
        """Synthesise speech for one utterance, blocking until done."""
        safe = str(text).replace('"', '\\"')

        if self.backend == "piper":
            cmd = (
                f'echo "{safe}" |'
                f' LD_LIBRARY_PATH={PIPER_LIB}'
                f' ESPEAK_DATA_PATH={PIPER_LIB}/espeak-ng-data'
                f' {PIPER_BIN} --model {PIPER_MODEL}'
                f' --output_raw | aplay -r 22050 -f S16_LE -c1 -D {AUDIO_DEVICE_LINUX}'
            )
        elif self.backend == "espeak_mbrola":
            cmd = (
                f'{ESPEAK_BIN} -v mb/mb-en1 -s {ESPEAK_SPEED} -p {ESPEAK_PITCH}'
                f' -g {ESPEAK_GAP} --stdout "{safe}" | aplay -D {AUDIO_DEVICE_LINUX}'
            )
        elif self.backend == "espeak":
            cmd = (
                f'{ESPEAK_BIN} -v {ESPEAK_VOICE} -s {ESPEAK_SPEED} -p {ESPEAK_PITCH}'
                f' -g {ESPEAK_GAP} --stdout "{safe}" | aplay -D {AUDIO_DEVICE_LINUX}'
            )
        else:
            cmd = None

        if cmd:
            proc = subprocess.Popen(cmd, shell=True)
            with self._proc_lock:
                self._proc = proc
            proc.wait()
            with self._proc_lock:
                self._proc = None
        elif self.backend == "pyttsx3":
            self._engine.say(text)
            self._engine.runAndWait()

    def _worker(self):
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
            # Clear queued messages and interrupt any currently playing speech
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._kill_current()
        self._q.put(text)

    def stop(self):
        self._stop.set()
        self._kill_current()


speech = SpeechManager()


# -------- Warning audio --------
def _get_warning_audio():
    base = os.path.dirname(os.path.abspath(__file__))
    for n in WARNING_AUDIO_CANDIDATES:
        p = os.path.join(base, n)
        if os.path.exists(p):
            return p
    return None


def _play_warning_once():
    global warning_process
    if warning_stop_event.is_set():
        return
    path = _get_warning_audio()
    if not path:
        time.sleep(1)
        return
    try:
        if sys.platform.startswith("linux"):
            warning_process = subprocess.Popen(
                ["aplay", "-D", AUDIO_DEVICE_LINUX, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            warning_process = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        time.sleep(0.2)
        return

    while warning_process.poll() is None:
        if warning_stop_event.is_set():
            try:
                warning_process.terminate()
            except Exception:
                pass
            break
        time.sleep(0.05)

    warning_process = None


def _warning_worker():
    while not warning_stop_event.is_set():
        _play_warning_once()


def start_warning_audio():
    global warning_active, warning_thread
    with warning_lock:
        if warning_active or (warning_thread and warning_thread.is_alive()):
            return
        warning_active = True
        warning_stop_event.clear()
        warning_thread = threading.Thread(target=_warning_worker, daemon=True)
        warning_thread.start()


def stop_warning_audio():
    global warning_active, warning_thread
    with warning_lock:
        if not warning_active and not (warning_thread and warning_thread.is_alive()):
            return
        warning_active = False
        warning_stop_event.set()
        if warning_process and warning_process.poll() is None:
            try:
                warning_process.terminate()
            except Exception:
                pass
    if warning_thread and warning_thread.is_alive():
        warning_thread.join(timeout=0.5)


# -------- Math helpers --------
def EAR(top, bottom, left, right):
    return np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)


def get_eye_features(lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    L_top = np.array(pts[LEFT_EYE_TOP])
    L_bottom = np.array(pts[LEFT_EYE_BOTTOM])
    L_left = np.array(pts[LEFT_EYE_LEFT])
    L_right = np.array(pts[LEFT_EYE_RIGHT])
    R_top = np.array(pts[RIGHT_EYE_TOP])
    R_bottom = np.array(pts[RIGHT_EYE_BOTTOM])
    R_left = np.array(pts[RIGHT_EYE_LEFT])
    R_right = np.array(pts[RIGHT_EYE_RIGHT])

    iris = np.array([pts[i] for i in IRIS_POINTS])
    midx = np.median(iris[:, 0])
    L_iris = iris[iris[:, 0] < midx].mean(axis=0)
    R_iris = iris[iris[:, 0] >= midx].mean(axis=0)

    EAR_L = EAR(L_top, L_bottom, L_left, L_right)
    EAR_R = EAR(R_top, R_bottom, R_left, R_right)

    L_off = (L_iris - (L_left + L_right) / 2) / np.linalg.norm(L_left - L_right)
    R_off = (R_iris - (R_left + R_right) / 2) / np.linalg.norm(R_left - R_right)

    return EAR_L, EAR_R, L_off, R_off


def get_head_offset(lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    face_c = (np.array(pts[FACE_LEFT]) + np.array(pts[FACE_RIGHT]) +
              np.array(pts[FACE_TOP]) + np.array(pts[FACE_BOTTOM])) / 4
    nose = np.array(pts[NOSE_TIP])
    fw = np.linalg.norm(np.array(pts[FACE_RIGHT]) - np.array(pts[FACE_LEFT])) + 1e-6
    fh = np.linalg.norm(np.array(pts[FACE_BOTTOM]) - np.array(pts[FACE_TOP])) + 1e-6
    return (nose[0] - face_c[0]) / fw, (nose[1] - face_c[1]) / fh


def get_face_metrics(lm, w, h):
    """Returns (face_width_ratio, center_x_ratio, center_y_ratio)."""
    pts = [(p.x * w, p.y * h) for p in lm]
    face_w = abs(pts[FACE_RIGHT][0] - pts[FACE_LEFT][0])
    cx = (pts[FACE_LEFT][0] + pts[FACE_RIGHT][0]) / 2
    cy = (pts[FACE_TOP][1] + pts[FACE_BOTTOM][1]) / 2
    return face_w / w, cx / w, cy / h


# -------- Camera --------
def open_camera():
    for i in (1, 2, 0, 3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        cap.release()
    return None


# -------- Calibration: Windows Hello-style guided pose --------
def _get_calib_hints(lm, w, h, EAR_L, EAR_R):
    """
    Returns a prioritised list of hints needed for the camera to see the face
    clearly. Empty list means the face is in a sampleable position.

    NOTE: We do NOT hint about head yaw/pitch — the user should be looking at
    their actual work. Calibration captures that natural pose as the baseline.
    Checks are ordered: distance → centering → eyes open.
    """
    hints = []
    face_w, cx, cy = get_face_metrics(lm, w, h)
    EDGE = 0.04  # 4% margin from frame edge

    # Face too small — camera too far to resolve features reliably
    if face_w < CALIB_FACE_SIZE_MIN:
        hints.append("Try moving the camera a little closer")

    # Face clipped at any edge — just needs to be fully in frame
    elif cx - face_w / 2 < EDGE:
        hints.append("Part of your face is cut off — shift the camera right")
    elif cx + face_w / 2 > 1.0 - EDGE:
        hints.append("Part of your face is cut off — shift the camera left")
    elif cy - face_w / 2 < EDGE:
        hints.append("Part of your face is cut off — angle the camera down")
    elif cy + face_w / 2 > 1.0 - EDGE:
        hints.append("Part of your face is cut off — angle the camera up")

    # Eyes must be open to establish a valid iris baseline
    if EAR_L < EAR_OPEN_THRESH or EAR_R < EAR_OPEN_THRESH:
        hints.append("Open your eyes fully so I can see them clearly")

    return hints


def calibrate(fm, cap):
    """
    Guided calibration. The timer only advances while the face is in a good
    position — resets whenever a hint fires. Returns (base_eye, base_yaw,
    base_pitch) on success, or None on failure.
    """
    speech.speak("calib_intro",
                 "Let's get you set up. Settle into your normal work position, just as you would naturally.",
                 cooldown=0, priority=True)
    time.sleep(3.0)
    speech.speak("calib_pos",
                 "Make sure your face is in the camera's view — you don't need to look at it.",
                 cooldown=0)

    eyes, yaws, pitches = [], [], []

    last_hint_text = ""
    last_hint_time = 0.0
    HINT_COOLDOWN = 2.5

    last_progress = -1
    good_start = None       # when the current good streak started
    bad_streak = 0          # consecutive bad frames — only reset after enough of these

    deadline = time.time() + 30.0   # hard deadline: give up after 30 s total

    while time.time() < deadline:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        r = fm.process(rgb)
        now = time.time()

        # ---- No face detected — always reset immediately ----
        if not r.multi_face_landmarks:
            bad_streak += 1
            if bad_streak >= CALIB_BAD_STREAK_RESET:
                good_start = None
                eyes.clear(); yaws.clear(); pitches.clear()
                last_progress = -1
            if now - last_hint_time > HINT_COOLDOWN:
                speech.speak("no_face_calib",
                             "I can't see your face yet. Try adjusting the camera angle until you're in frame.",
                             cooldown=0, priority=True)
                last_hint_text = "no_face"
                last_hint_time = now
            continue

        lm = r.multi_face_landmarks[0].landmark
        EAR_L, EAR_R, L, R = get_eye_features(lm, *f.shape[1::-1])
        hints = _get_calib_hints(lm, *f.shape[1::-1], EAR_L, EAR_R)

        # ---- Face needs adjustment ----
        if hints:
            bad_streak += 1
            hint = hints[0]
            if bad_streak >= CALIB_BAD_STREAK_RESET:
                good_start = None
                eyes.clear(); yaws.clear(); pitches.clear()
                last_progress = -1
            if hint != last_hint_text and now - last_hint_time > HINT_COOLDOWN:
                speech.speak(f"hint_{hint}", hint, cooldown=0, priority=True)
                last_hint_text = hint
                last_hint_time = now
            continue

        # ---- Face is good — start / continue sampling ----
        bad_streak = 0
        if good_start is None:
            good_start = now
            speech.speak("hold_still", "Perfect. Stay just like that.", cooldown=0)
            last_hint_text = "hold_still"
            last_hint_time = now

        elapsed_good = now - good_start
        eye = (L + R) / 2
        yaw, pitch = get_head_offset(lm, *f.shape[1::-1])
        eyes.append(eye)
        yaws.append(yaw)
        pitches.append(pitch)

        # Progress milestones
        progress = int((elapsed_good / CALIB_SECONDS) * 100)
        _PROGRESS_LINES = {25: "Almost there, keep going.", 50: "Halfway through.", 75: "Nearly done."}
        for milestone, line in _PROGRESS_LINES.items():
            if last_progress < milestone <= progress:
                speech.speak(f"prog_{milestone}", line, cooldown=0)
        last_progress = progress

        if elapsed_good >= CALIB_SECONDS:
            break

    if len(eyes) < CALIB_MIN_SAMPLES:
        speech.speak("calib_fail",
                     "That didn't quite work. Make sure your face is in view and we'll try again.",
                     cooldown=0, priority=True)
        return None

    speech.speak("calib_done",
                 "All set. I've learned your natural working position and I'll keep an eye on things from here.",
                 cooldown=0, priority=True)
    return np.median(eyes, 0), np.median(yaws), np.median(pitches)


# -------- Per-frame focus classification --------
def classify_frame(lm, w, h, base_eye, base_yaw, base_pitch, yaw=None, pitch=None):
    """Returns (is_focused: bool, FocusState).
    Pass pre-smoothed yaw/pitch to apply dead band filtering."""
    EAR_L, EAR_R, L, R = get_eye_features(lm, w, h)
    if yaw is None or pitch is None:
        yaw, pitch = get_head_offset(lm, w, h)
    eye = (L + R) / 2

    eyes_open = EAR_L > EAR_OPEN_THRESH and EAR_R > EAR_OPEN_THRESH
    head_ok = (abs(yaw - base_yaw) < HEAD_YAW_THRESH and
               abs(pitch - base_pitch) < HEAD_PITCH_THRESH)
    eye_ok = np.linalg.norm(eye - base_eye) < EYE_REL_THRESH

    if not eyes_open:
        return False, FocusState.DROWSY
    if not head_ok:
        return False, FocusState.HEAD_TURNED
    if not eye_ok:
        return False, FocusState.LOOKING_AWAY
    return True, FocusState.FOCUSED


# -------- Warning state machine --------
class WarningController:
    """
    Hysteresis-based state machine:
      FOCUSED  → WARNING  when bad_ratio >= WARN_ENTER_RATIO
      WARNING  → FOCUSED  when bad_ratio <  WARN_EXIT_RATIO

    While in WARNING, speech messages escalate after WARN_ESCALATE_SEC seconds
    with per-state targeted feedback (drowsy / looking away / head turned).
    """

    _MESSAGES = {
        FocusState.DROWSY: (
            "Hey, your eyes look heavy. Try to stay alert.",
            "You've been drowsy for a while now. It might be time to take a short break.",
        ),
        FocusState.LOOKING_AWAY: (
            "Looks like you've drifted — come back to your work when you're ready.",
            "You've been away from your work for quite a while. Time to refocus.",
        ),
        FocusState.HEAD_TURNED: (
            "Something caught your attention — ready to get back to it?",
            "You've been turned away for a while now. Let's get back on track.",
        ),
    }
    _DEFAULT_MESSAGES = (
        "Hey, looks like you got a bit distracted. Let's get back on track.",
        "You've been unfocused for a while. Try to bring your attention back to your work.",
    )

    def __init__(self):
        self._in_warning = False
        self._warn_start = None

    def update(self, bad_ratio: float, dominant_state: FocusState, no_face: bool):
        if no_face:
            # No-face is handled separately; clear audio warning if active.
            if self._in_warning:
                self._exit_warning(speak_recovery=False)
            return

        if not self._in_warning:
            if bad_ratio >= WARN_ENTER_RATIO:
                self._in_warning = True
                self._warn_start = time.time()
                self._emit_warning(dominant_state)
        else:
            if bad_ratio < WARN_EXIT_RATIO:
                self._exit_warning(speak_recovery=True)
            else:
                self._emit_warning(dominant_state)

    def _emit_warning(self, state: FocusState):
        start_warning_audio()
        escalated = (time.time() - self._warn_start) >= WARN_ESCALATE_SEC
        gentle, urgent = self._MESSAGES.get(state, self._DEFAULT_MESSAGES)
        text = urgent if escalated else gentle
        key = f"warn_{state.value}_{'esc' if escalated else 'soft'}"
        cooldown = 8.0 if escalated else 12.0
        speech.speak(key, text, cooldown)

    def _exit_warning(self, speak_recovery: bool):
        self._in_warning = False
        self._warn_start = None
        stop_warning_audio()
        if speak_recovery:
            speech.speak("refocused", "Welcome back — you're focused again. Keep it up.", cooldown=0)

    @property
    def in_warning(self):
        return self._in_warning


# -------- Main --------
def main():
    cap = open_camera()
    if cap is None:
        raise SystemExit("Camera not found")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    focus_history = deque()
    state_history = deque()
    no_person_since = None
    warning_ctrl = WarningController()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:

        # Calibration: retry automatically until successful
        calib_result = None
        while calib_result is None:
            calib_result = calibrate(fm, cap)
            if calib_result is None:
                speech.speak("retry_calib",
                             "Let's give it another try in a moment.",
                             cooldown=0, priority=True)
                time.sleep(3.0)

        base_eye, base_yaw, base_pitch = calib_result
        smooth_yaw   = base_yaw
        smooth_pitch = base_pitch

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                now = time.time()
                cutoff = now - FOCUS_WINDOW_SECONDS

                if res.multi_face_landmarks:
                    no_person_since = None
                    lm = res.multi_face_landmarks[0].landmark

                    raw_yaw, raw_pitch = get_head_offset(lm, w, h)
                    smooth_yaw   = YAW_SMOOTH_ALPHA   * raw_yaw   + (1 - YAW_SMOOTH_ALPHA)   * smooth_yaw
                    smooth_pitch = PITCH_SMOOTH_ALPHA * raw_pitch + (1 - PITCH_SMOOTH_ALPHA) * smooth_pitch

                    focused, frame_state = classify_frame(
                        lm, w, h, base_eye, base_yaw, base_pitch,
                        yaw=smooth_yaw, pitch=smooth_pitch
                    )

                    focus_history.append((now, focused))
                    state_history.append((now, frame_state))

                    # Trim to rolling window
                    while focus_history and focus_history[0][0] < cutoff:
                        focus_history.popleft()
                    while state_history and state_history[0][0] < cutoff:
                        state_history.popleft()

                    bad_ratio = (
                        sum(not x for _, x in focus_history)
                        / max(1, len(focus_history))
                    )

                    # Find the most common non-focused state for targeted messages
                    bad_states = [s for _, s in state_history if s != FocusState.FOCUSED]
                    dominant = (
                        Counter(bad_states).most_common(1)[0][0]
                        if bad_states else FocusState.FOCUSED
                    )

                    warning_ctrl.update(bad_ratio, dominant, no_face=False)

                else:
                    # No face in frame
                    if no_person_since is None:
                        no_person_since = now

                    focus_history.clear()
                    state_history.clear()
                    warning_ctrl.update(0.0, FocusState.NO_FACE, no_face=True)

                    if now - no_person_since >= NO_FACE_PROMPT_SEC:
                        speech.speak(
                            "no_person",
                            "I can't see you anymore. Make sure you're still in the camera's view.",
                            cooldown=5.0,
                            priority=True,
                        )

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass

    stop_warning_audio()
    speech.stop()
    cap.release()


if __name__ == "__main__":
    main()
