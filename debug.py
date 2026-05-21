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

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

mp_face_mesh = mp.solutions.face_mesh

# -------- FaceMesh indices --------
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 263
RIGHT_EYE_RIGHT  = 362
IRIS_POINTS = list(range(468, 478))
FACE_LEFT   = 234
FACE_RIGHT  = 454
FACE_TOP    = 10
FACE_BOTTOM = 152
NOSE_TIP    = 1

# -------- Calibration thresholds --------
CALIB_SECONDS       = 5.0
CALIB_MIN_SAMPLES   = 30
CALIB_FACE_SIZE_MIN = 0.20  # face_width / frame_width — minimum to resolve features

# -------- Calibration streak --------
CALIB_BAD_STREAK_RESET = 10  # consecutive bad frames before resetting progress

# -------- Detection thresholds --------
EAR_OPEN_THRESH   = 0.20
EYE_REL_THRESH    = 0.12
HEAD_YAW_THRESH   = 0.15
HEAD_PITCH_THRESH = 0.15

# -------- Pose smoothing (EMA) — dead band for minor head movements --------
YAW_SMOOTH_ALPHA   = 0.15
PITCH_SMOOTH_ALPHA = 0.15

# -------- Warning state machine --------
FOCUS_WINDOW_SECONDS = 10.0
WARN_ENTER_RATIO     = 0.60
WARN_EXIT_RATIO      = 0.30
WARN_ESCALATE_SEC    = 15.0

NO_FACE_PROMPT_SEC       = 2.0
WARNING_AUDIO_CANDIDATES = ["warning.wav", "warning.mp3", "warning.aiff"]

# -------- Window layout --------
CAM_W   = 640
CAM_H   = 480
PANEL_W = 320
WIN_W   = CAM_W + PANEL_W
WIN_H   = CAM_H

# -------- Colors (BGR) --------
C_BG     = (30, 30, 46)
C_PANEL  = (45, 45, 65)
C_GREEN  = (80, 200, 120)
C_YELLOW = (40, 200, 220)
C_ORANGE = (60, 140, 255)
C_RED    = (80, 80, 220)
C_WHITE  = (230, 230, 230)
C_GRAY   = (120, 120, 140)
C_DARK   = (20, 20, 35)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# -------- Focus states --------
class FocusState(Enum):
    FOCUSED      = "focused"
    DROWSY       = "drowsy"
    LOOKING_AWAY = "looking_away"
    HEAD_TURNED  = "head_turned"
    NO_FACE      = "no_face"

STATE_COLOR = {
    FocusState.FOCUSED:      C_GREEN,
    FocusState.DROWSY:       C_YELLOW,
    FocusState.LOOKING_AWAY: C_ORANGE,
    FocusState.HEAD_TURNED:  C_ORANGE,
    FocusState.NO_FACE:      C_GRAY,
}
STATE_LABEL = {
    FocusState.FOCUSED:      "Focused",
    FocusState.DROWSY:       "Drowsy",
    FocusState.LOOKING_AWAY: "Looking Away",
    FocusState.HEAD_TURNED:  "Head Turned",
    FocusState.NO_FACE:      "No Face",
}


# -------- Warning audio --------
warning_active    = False
warning_stop_event = threading.Event()
warning_process   = None
warning_thread    = None
warning_lock      = threading.Lock()


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
        if sys.platform == "darwin":
            warning_process = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while warning_process.poll() is None:
                if warning_stop_event.is_set():
                    try:
                        warning_process.terminate()
                    except Exception:
                        pass
                    break
                time.sleep(0.05)
            warning_process = None
        elif sys.platform == "win32":
            if HAS_WINSOUND and path.endswith(".wav"):
                # SND_FILENAME is blocking — runs inside the daemon thread
                winsound.PlaySound(path, winsound.SND_FILENAME)
            else:
                # Fallback: audible beep
                winsound.Beep(880, 500)
                time.sleep(0.5)
        else:
            time.sleep(1)
    except Exception:
        time.sleep(0.2)


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
        if sys.platform == "win32" and HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
    if warning_thread and warning_thread.is_alive():
        warning_thread.join(timeout=0.5)


# -------- Speech (pyttsx3 — Mac/Windows) --------
class SpeechManager:
    def __init__(self):
        self._q      = queue.Queue()
        self._last   = {}
        self._stop   = threading.Event()
        self._engine = None
        self.enabled = pyttsx3 is not None

        if self.enabled:
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
        """Interrupt currently playing speech."""
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass

    def _worker(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._engine.say(text)
                self._engine.runAndWait()
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

    def stop(self):
        self._stop.set()
        self._kill_current()


speech = SpeechManager()


# -------- Math helpers --------
def EAR(top, bottom, left, right):
    return np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)


def get_eye_features(lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    L_top    = np.array(pts[LEFT_EYE_TOP])
    L_bottom = np.array(pts[LEFT_EYE_BOTTOM])
    L_left   = np.array(pts[LEFT_EYE_LEFT])
    L_right  = np.array(pts[LEFT_EYE_RIGHT])
    R_top    = np.array(pts[RIGHT_EYE_TOP])
    R_bottom = np.array(pts[RIGHT_EYE_BOTTOM])
    R_left   = np.array(pts[RIGHT_EYE_LEFT])
    R_right  = np.array(pts[RIGHT_EYE_RIGHT])
    iris = np.array([pts[i] for i in IRIS_POINTS])
    midx   = np.median(iris[:, 0])
    L_iris = iris[iris[:, 0] < midx].mean(axis=0)
    R_iris = iris[iris[:, 0] >= midx].mean(axis=0)
    EAR_L = EAR(L_top, L_bottom, L_left, L_right)
    EAR_R = EAR(R_top, R_bottom, R_left, R_right)
    L_off = (L_iris - (L_left + L_right) / 2) / np.linalg.norm(L_left - L_right)
    R_off = (R_iris - (R_left + R_right) / 2) / np.linalg.norm(R_left - R_right)
    return EAR_L, EAR_R, L_off, R_off


def get_head_offset(lm, w, h):
    pts    = [(int(p.x * w), int(p.y * h)) for p in lm]
    face_c = (np.array(pts[FACE_LEFT]) + np.array(pts[FACE_RIGHT]) +
              np.array(pts[FACE_TOP])  + np.array(pts[FACE_BOTTOM])) / 4
    nose   = np.array(pts[NOSE_TIP])
    fw = np.linalg.norm(np.array(pts[FACE_RIGHT]) - np.array(pts[FACE_LEFT])) + 1e-6
    fh = np.linalg.norm(np.array(pts[FACE_BOTTOM]) - np.array(pts[FACE_TOP]))  + 1e-6
    return (nose[0] - face_c[0]) / fw, (nose[1] - face_c[1]) / fh


def get_face_metrics(lm, w, h):
    pts    = [(p.x * w, p.y * h) for p in lm]
    face_w = abs(pts[FACE_RIGHT][0] - pts[FACE_LEFT][0])
    cx     = (pts[FACE_LEFT][0] + pts[FACE_RIGHT][0]) / 2
    cy     = (pts[FACE_TOP][1]  + pts[FACE_BOTTOM][1]) / 2
    return face_w / w, cx / w, cy / h


# -------- Camera --------
def open_camera():
    for i in (0, 1, 2, 3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        cap.release()
    return None


# -------- Calibration hints (no head-pose hints — captures natural pose) --------
def _get_calib_hints(lm, w, h, EAR_L, EAR_R):
    hints  = []
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


# -------- Per-frame focus classification --------
def classify_frame(lm, w, h, base_eye, base_yaw, base_pitch, yaw=None, pitch=None):
    """Returns (is_focused: bool, FocusState).
    Pass pre-smoothed yaw/pitch to apply dead band filtering."""
    EAR_L, EAR_R, L, R = get_eye_features(lm, w, h)
    if yaw is None or pitch is None:
        yaw, pitch = get_head_offset(lm, w, h)
    eye = (L + R) / 2

    eyes_open = EAR_L > EAR_OPEN_THRESH and EAR_R > EAR_OPEN_THRESH
    head_ok   = (abs(yaw - base_yaw)   < HEAD_YAW_THRESH and
                 abs(pitch - base_pitch) < HEAD_PITCH_THRESH)
    eye_ok    = np.linalg.norm(eye - base_eye) < EYE_REL_THRESH

    if not eyes_open:
        return False, FocusState.DROWSY
    if not head_ok:
        return False, FocusState.HEAD_TURNED
    if not eye_ok:
        return False, FocusState.LOOKING_AWAY
    return True, FocusState.FOCUSED


# -------- Warning state machine --------
class WarningController:
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
        text     = urgent if escalated else gentle
        key      = f"warn_{state.value}_{'esc' if escalated else 'soft'}"
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


# -------- Drawing helpers --------
def draw_text(img, text, pos, scale=0.52, color=C_WHITE, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bar(img, rect, value, max_value, fg_color, bg_color=C_DARK):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill = int(w * min(value / max(max_value, 1e-6), 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_GRAY, 1)


def draw_section(canvas, label, y, px):
    draw_text(canvas, label.upper(), (px, y), scale=0.38, color=C_GRAY)
    cv2.line(canvas, (px, y + 6), (WIN_W - 20, y + 6), C_GRAY, 1)
    return y + 18


def wrap_text(text, max_chars=26):
    words  = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            if line:
                lines.append(line.rstrip())
            line = w + " "
        else:
            line += w + " "
    if line.strip():
        lines.append(line.strip())
    return lines


# -------- Calibration screen --------
def _calibration_loop(fm, cap):
    """
    Visual guided calibration. Draws an oval guide on the camera feed and a
    hint panel beside it. Timer only advances while face is in a good position.
    Returns (base_eye, base_yaw, base_pitch) or None on failure/quit.
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
    HINT_COOLDOWN  = 2.5
    last_progress  = -1
    good_start     = None
    bad_streak     = 0           # consecutive bad frames before resetting progress
    deadline       = time.time() + 30.0
    calib_state    = "waiting"   # waiting | hint | sampling
    current_hint   = ""

    while time.time() < deadline:
        ok, raw = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        canvas = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)
        feed   = cv2.resize(raw, (CAM_W, CAM_H))
        canvas[:, :CAM_W] = feed

        now = time.time()
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        r   = fm.process(rgb)

        # Guide oval
        oval_cx, oval_cy = CAM_W // 2, CAM_H // 2
        oval_rx, oval_ry = 130, 165
        oval_color = {
            "waiting":  C_GRAY,
            "hint":     C_ORANGE,
            "sampling": C_GREEN,
        }.get(calib_state, C_GRAY)
        cv2.ellipse(canvas, (oval_cx, oval_cy), (oval_rx, oval_ry),
                    0, 0, 360, oval_color, 3)

        # Progress arc around oval
        if calib_state == "sampling" and good_start is not None:
            elapsed = now - good_start
            angle   = int(min(elapsed / CALIB_SECONDS, 1.0) * 360)
            cv2.ellipse(canvas, (oval_cx, oval_cy), (oval_rx + 12, oval_ry + 12),
                        -90, 0, angle, C_GREEN, 5)

        # Panel header
        px = CAM_W + 20
        draw_text(canvas, "CALIBRATION", (px, 35), scale=0.65, color=C_WHITE, thickness=2)
        draw_text(canvas, "Set up your working position", (px, 58), scale=0.40, color=C_GRAY)
        cv2.line(canvas, (px, 70), (WIN_W - 20, 70), C_GRAY, 1)

        if not r.multi_face_landmarks:
            bad_streak += 1
            if bad_streak >= CALIB_BAD_STREAK_RESET:
                calib_state = "waiting"
                good_start  = None
                eyes.clear(); yaws.clear(); pitches.clear()
                last_progress = -1
            if now - last_hint_time > HINT_COOLDOWN:
                speech.speak("no_face_calib",
                             "I can't see your face yet. Try adjusting the camera angle until you're in frame.",
                             cooldown=0, priority=True)
                last_hint_text = "no_face"
                last_hint_time = now
                current_hint   = "No face detected"

            draw_text(canvas, "No face detected", (px, 100), scale=0.55, color=C_RED)
            draw_text(canvas, "Adjust the camera angle", (px, 126), scale=0.42, color=C_GRAY)

        else:
            lm     = r.multi_face_landmarks[0].landmark
            EAR_L, EAR_R, L, R = get_eye_features(lm, raw.shape[1], raw.shape[0])
            hints  = _get_calib_hints(lm, raw.shape[1], raw.shape[0], EAR_L, EAR_R)

            # Landmark dots on feed
            key_pts = [LEFT_EYE_LEFT, LEFT_EYE_RIGHT, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
                       FACE_LEFT, FACE_RIGHT, NOSE_TIP]
            for idx in key_pts:
                p  = lm[idx]
                px_ = int(p.x * CAM_W)
                py_ = int(p.y * CAM_H)
                cv2.circle(canvas, (px_, py_), 4, oval_color, -1)

            if hints:
                bad_streak += 1
                calib_state = "hint"
                hint = hints[0]
                current_hint = hint
                if bad_streak >= CALIB_BAD_STREAK_RESET:
                    good_start    = None
                    last_progress = -1
                    eyes.clear(); yaws.clear(); pitches.clear()
                if hint != last_hint_text and now - last_hint_time > HINT_COOLDOWN:
                    speech.speak(f"hint_{hint}", hint, cooldown=0, priority=True)
                    last_hint_text = hint
                    last_hint_time = now

                draw_text(canvas, "Adjustment needed", (px, 100), scale=0.55, color=C_ORANGE)
                cv2.line(canvas, (px, 115), (WIN_W - 20, 115), C_PANEL, 1)
                for i, ln in enumerate(wrap_text(hint)):
                    draw_text(canvas, ln, (px, 140 + i * 26), scale=0.50, color=C_WHITE)

            else:
                bad_streak  = 0
                calib_state = "sampling"
                if good_start is None:
                    good_start = now
                    speech.speak("hold_still", "Perfect. Stay just like that.", cooldown=0)
                    last_hint_text = "hold_still"
                    last_hint_time = now

                elapsed = now - good_start
                eye     = (L + R) / 2
                yaw, pitch = get_head_offset(lm, raw.shape[1], raw.shape[0])
                eyes.append(eye); yaws.append(yaw); pitches.append(pitch)

                progress = int((elapsed / CALIB_SECONDS) * 100)
                for milestone, line in [(25, "Almost there, keep going."),
                                        (50, "Halfway through."),
                                        (75, "Nearly done.")]:
                    if last_progress < milestone <= progress:
                        speech.speak(f"prog_{milestone}", line, cooldown=0)
                last_progress = progress

                draw_text(canvas, "Recording pose...", (px, 100), scale=0.55, color=C_GREEN)
                cv2.line(canvas, (px, 115), (WIN_W - 20, 115), C_PANEL, 1)
                draw_text(canvas, "Stay in your natural", (px, 142), scale=0.43, color=C_GRAY)
                draw_text(canvas, "working position",     (px, 164), scale=0.43, color=C_GRAY)

                draw_text(canvas, "Progress", (px, 200), scale=0.42, color=C_GRAY)
                draw_bar(canvas, (px, 210, PANEL_W - 40, 18), progress, 100, C_GREEN)
                draw_text(canvas, f"{progress}%",
                          (px + (PANEL_W - 40) // 2 - 14, 224), scale=0.40, color=C_WHITE)

                if elapsed >= CALIB_SECONDS:
                    break

        draw_text(canvas, "ESC  quit", (px, WIN_H - 20), scale=0.38, color=C_GRAY)
        cv2.imshow("Focus Monitor", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return None

    if len(eyes) < CALIB_MIN_SAMPLES:
        speech.speak("calib_fail",
                     "That didn't quite work. Make sure your face is in view and we'll try again.",
                     cooldown=0, priority=True)
        return None

    speech.speak("calib_done",
                 "All set. I've learned your natural working position and I'll keep an eye on things from here.",
                 cooldown=0, priority=True)
    return np.median(eyes, 0), np.median(yaws), np.median(pitches)


# -------- Monitoring screen --------
def draw_monitor(canvas, frame, frame_state, bad_ratio,
                 EAR_L, EAR_R, yaw, pitch, base_yaw, base_pitch,
                 in_warning, focus_history):
    # Camera feed
    canvas[:, :CAM_W] = cv2.resize(frame, (CAM_W, CAM_H))

    # Flashing red/orange border when warning is active
    if in_warning:
        flash  = int(time.time() * 2) % 2 == 0
        border = C_RED if flash else C_ORANGE
        cv2.rectangle(canvas, (0, 0), (CAM_W - 1, CAM_H - 1), border, 8)

    # Panel
    canvas[:, CAM_W:] = C_PANEL
    px = CAM_W + 20
    y  = 30

    # Title
    draw_text(canvas, "FOCUS MONITOR", (px, y), scale=0.65, color=C_WHITE, thickness=2)
    y += 12
    cv2.line(canvas, (px, y + 6), (WIN_W - 20, y + 6), C_GRAY, 1)
    y += 22

    # ---- Status ----
    state_color = C_RED if in_warning else STATE_COLOR.get(frame_state, C_GRAY)
    label       = ("WARNING" if in_warning else STATE_LABEL.get(frame_state, "Unknown")).upper()
    cv2.circle(canvas, (px + 9, y + 5), 9, state_color, -1)
    draw_text(canvas, label, (px + 26, y + 11),
              scale=0.62, color=state_color,
              thickness=2 if in_warning else 1)
    y += 32
    cv2.line(canvas, (px, y), (WIN_W - 20, y), C_GRAY, 1)
    y += 14

    # ---- Focus score ----
    y = draw_section(canvas, "Focus Score", y, px)
    focus_score = 1.0 - bad_ratio
    score_color = C_GREEN if focus_score > 0.7 else (C_YELLOW if focus_score > 0.4 else C_RED)
    draw_text(canvas, f"{int(focus_score * 100)}%", (px, y + 14), scale=0.55, color=score_color)
    draw_bar(canvas, (px + 38, y, PANEL_W - 60, 16), focus_score, 1.0, score_color)
    y += 30

    # ---- Rolling history mini bar chart ----
    history_list = list(focus_history)
    n = len(history_list)
    if n > 0:
        bar_w = max((PANEL_W - 40) // n, 2)
        for i, (_, fok) in enumerate(history_list):
            bx = px + i * bar_w
            cv2.rectangle(canvas, (bx, y), (bx + bar_w - 1, y + 10),
                          C_GREEN if fok else C_RED, -1)
    y += 18
    draw_text(canvas, "10-second window", (px, y), scale=0.36, color=C_GRAY)
    y += 18
    cv2.line(canvas, (px, y), (WIN_W - 20, y), C_GRAY, 1)
    y += 14

    # ---- Eyes ----
    y = draw_section(canvas, "Eyes", y, px)
    for side, ear in [("L", EAR_L), ("R", EAR_R)]:
        open_c = C_GREEN if ear > EAR_OPEN_THRESH else C_RED
        label_ = "Open" if ear > EAR_OPEN_THRESH else "Closed"
        draw_text(canvas, f"{side}  {label_}  ({ear:.2f})", (px, y), scale=0.44, color=open_c)
        y += 20
    y += 4
    cv2.line(canvas, (px, y), (WIN_W - 20, y), C_GRAY, 1)
    y += 14

    # ----  Head pose deviation ----
    y = draw_section(canvas, "Head Pose Deviation", y, px)
    bar_total = PANEL_W - 40
    for lbl, val, thresh in [("Yaw",   yaw   - base_yaw,   HEAD_YAW_THRESH),
                               ("Pitch", pitch - base_pitch, HEAD_PITCH_THRESH)]:
        within    = abs(val) < thresh
        bar_color = C_GREEN if within else C_RED
        draw_text(canvas, lbl, (px, y + 12), scale=0.40, color=C_GRAY)
        bx  = px + 45
        bw  = bar_total - 45
        mid = bx + bw // 2
        norm   = max(min(val / (thresh * 2), 1.0), -1.0)
        fill_x = int(mid + norm * bw / 2)
        cv2.rectangle(canvas, (bx, y), (bx + bw, y + 14), C_DARK, -1)
        if norm >= 0:
            cv2.rectangle(canvas, (mid, y), (fill_x, y + 14), bar_color, -1)
        else:
            cv2.rectangle(canvas, (fill_x, y), (mid, y + 14), bar_color, -1)
        cv2.line(canvas, (mid, y - 2), (mid, y + 16), C_GRAY, 1)
        cv2.rectangle(canvas, (bx, y), (bx + bw, y + 14), C_GRAY, 1)
        y += 22
    y += 4
    cv2.line(canvas, (px, y), (WIN_W - 20, y), C_GRAY, 1)
    y += 14

    # ---- Distraction level with threshold markers ----
    y = draw_section(canvas, "Distraction Level", y, px)
    dist_color = (C_RED if bad_ratio >= WARN_ENTER_RATIO
                  else C_YELLOW if bad_ratio >= WARN_EXIT_RATIO
                  else C_GREEN)
    draw_bar(canvas, (px, y, bar_total, 16), bad_ratio, 1.0, dist_color)
    enter_x = px + int(bar_total * WARN_ENTER_RATIO)
    exit_x  = px + int(bar_total * WARN_EXIT_RATIO)
    cv2.line(canvas, (enter_x, y - 3), (enter_x, y + 19), C_RED,   2)
    cv2.line(canvas, (exit_x,  y - 3), (exit_x,  y + 19), C_GREEN, 2)
    y += 24
    draw_text(canvas, f"{int(bad_ratio * 100)}% distracted", (px, y),
              scale=0.40, color=dist_color)

    draw_text(canvas, "ESC  quit", (px, WIN_H - 20), scale=0.38, color=C_GRAY)


# -------- Main --------
def main():
    cap = open_camera()
    if cap is None:
        raise SystemExit("Camera not found")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Focus Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Focus Monitor", WIN_W, WIN_H)

    focus_history  = deque()
    state_history  = deque()
    no_person_since = None
    warning_ctrl   = WarningController()

    # Persistent display values (carry last known across no-face frames)
    EAR_L = EAR_R = 0.3
    yaw = pitch = base_yaw = base_pitch = 0.0
    frame_state = FocusState.NO_FACE
    bad_ratio   = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:

        # Calibration — retry until success or ESC
        calib_result = None
        while calib_result is None:
            calib_result = _calibration_loop(fm, cap)
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
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res  = fm.process(rgb)
                now  = time.time()
                cutoff = now - FOCUS_WINDOW_SECONDS

                canvas = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)

                if res.multi_face_landmarks:
                    no_person_since = None
                    lm = res.multi_face_landmarks[0].landmark

                    EAR_L, EAR_R, _, _ = get_eye_features(lm, w, h)
                    yaw, pitch          = get_head_offset(lm, w, h)
                    smooth_yaw   = YAW_SMOOTH_ALPHA   * yaw   + (1 - YAW_SMOOTH_ALPHA)   * smooth_yaw
                    smooth_pitch = PITCH_SMOOTH_ALPHA * pitch + (1 - PITCH_SMOOTH_ALPHA) * smooth_pitch

                    focused, frame_state = classify_frame(
                        lm, w, h, base_eye, base_yaw, base_pitch,
                        yaw=smooth_yaw, pitch=smooth_pitch
                    )

                    focus_history.append((now, focused))
                    state_history.append((now, frame_state))

                    while focus_history and focus_history[0][0] < cutoff:
                        focus_history.popleft()
                    while state_history and state_history[0][0] < cutoff:
                        state_history.popleft()

                    bad_ratio = (
                        sum(not x for _, x in focus_history)
                        / max(1, len(focus_history))
                    )
                    bad_states = [s for _, s in state_history if s != FocusState.FOCUSED]
                    dominant   = (Counter(bad_states).most_common(1)[0][0]
                                  if bad_states else FocusState.FOCUSED)
                    warning_ctrl.update(bad_ratio, dominant, no_face=False)

                else:
                    frame_state = FocusState.NO_FACE
                    if no_person_since is None:
                        no_person_since = now
                    focus_history.clear()
                    state_history.clear()
                    bad_ratio = 0.0
                    warning_ctrl.update(0.0, FocusState.NO_FACE, no_face=True)
                    if now - no_person_since >= NO_FACE_PROMPT_SEC:
                        speech.speak(
                            "no_person",
                            "I can't see you anymore. Make sure you're still in the camera's view.",
                            cooldown=5.0,
                            priority=True,
                        )

                draw_monitor(canvas, frame, frame_state, bad_ratio,
                             EAR_L, EAR_R, yaw, pitch, base_yaw, base_pitch,
                             warning_ctrl.in_warning, focus_history)

                cv2.imshow("Focus Monitor", canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            pass

    stop_warning_audio()
    speech.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
