import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import subprocess
import threading
import queue
from collections import deque

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

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

# -------- Calibration & thresholds --------
CALIB_SECONDS = 5.0
CALIB_MIN_SAMPLES = 20

EAR_OPEN_THRESH = 0.20
EYE_REL_THRESH = 0.12
HEAD_YAW_THRESH = 0.15
HEAD_PITCH_THRESH = 0.15

FOCUS_WINDOW_SECONDS = 10.0
NOT_FOCUS_THRESHOLD = 0.8

NO_FACE_PROMPT_SEC = 2.0

AUDIO_DEVICE_LINUX = "plughw:1,0"
WARNING_AUDIO_CANDIDATES = ["warning.wav", "warning.mp3", "warning.aiff"]

# -------- warning.wav control --------
warning_active = False
warning_stop_event = threading.Event()
warning_process = None


# -------- Speech --------
class SpeechManager:
    def __init__(self):
        self.enabled = pyttsx3 is not None
        self._q = queue.Queue()
        self._last = {}
        self._stop = threading.Event()

        if self.enabled:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 180)
            except Exception:
                self.enabled = False

        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

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
                self.enabled = False

    def speak(self, key, text, cooldown=3.0):
        if not self.enabled:
            return
        now = time.time()
        if now - self._last.get(key, 0) < cooldown:
            return
        self._last[key] = now
        while not self._q.empty():
            self._q.get_nowait()
        self._q.put(text)

    def stop(self):
        self._stop.set()


speech = SpeechManager()


# -------- warning audio --------
def _get_warning_audio():
    base = os.path.dirname(os.path.abspath(__file__))
    for n in WARNING_AUDIO_CANDIDATES:
        p = os.path.join(base, n)
        if os.path.exists(p):
            return p
    return None


def _play_warning_once():
    global warning_process
    path = _get_warning_audio()
    if not path:
        time.sleep(1)
        return

    try:
        if sys.platform.startswith("linux"):
            warning_process = subprocess.Popen(
                ["aplay", "-D", AUDIO_DEVICE_LINUX, path]
            )
        else:
            warning_process = subprocess.Popen(["afplay", path])
    except Exception:
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
    global warning_active
    if warning_active:
        return
    warning_active = True
    warning_stop_event.clear()
    threading.Thread(target=_warning_worker, daemon=True).start()


def stop_warning_audio():
    global warning_active
    if not warning_active:
        return
    warning_active = False
    warning_stop_event.set()
    if warning_process and warning_process.poll() is None:
        try:
            warning_process.terminate()
        except Exception:
            pass


# -------- math helpers --------
def EAR(top, bottom, left, right):
    return np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)


def get_eye_features(lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    L_top, L_bottom = np.array(pts[LEFT_EYE_TOP]), np.array(pts[LEFT_EYE_BOTTOM])
    L_left, L_right = np.array(pts[LEFT_EYE_LEFT]), np.array(pts[LEFT_EYE_RIGHT])
    R_top, R_bottom = np.array(pts[RIGHT_EYE_TOP]), np.array(pts[RIGHT_EYE_BOTTOM])
    R_left, R_right = np.array(pts[RIGHT_EYE_LEFT]), np.array(pts[RIGHT_EYE_RIGHT])

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


# -------- camera --------
def open_camera():
    for i in (1, 2, 0, 3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def calibrate(fm, cap):
    speech.speak("calib", "Calibration starting", 0)
    eyes, yaws, pitches = [], [], []

    t0 = time.time()
    while time.time() - t0 < CALIB_SECONDS:
        ok, f = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        r = fm.process(rgb)
        if not r.multi_face_landmarks:
            continue

        lm = r.multi_face_landmarks[0].landmark
        EAR_L, EAR_R, L, R = get_eye_features(lm, *f.shape[1::-1])
        if EAR_L < EAR_OPEN_THRESH or EAR_R < EAR_OPEN_THRESH:
            continue

        eye = (L + R) / 2
        yaw, pitch = get_head_offset(lm, *f.shape[1::-1])

        eyes.append(eye)
        yaws.append(yaw)
        pitches.append(pitch)

    if len(eyes) < CALIB_MIN_SAMPLES:
        speech.speak("fail", "Calibration failed", 0)
        return np.zeros(2), 0.0, 0.0

    speech.speak("done", "Calibration completed", 0)
    return np.median(eyes, 0), np.median(yaws), np.median(pitches)


# -------- main --------
def main():
    cap = open_camera()
    if cap is None:
        raise SystemExit("Camera not found")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    focus_history = deque()
    no_person_since = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:

        stop_warning_audio()
        base_eye, base_yaw, base_pitch = calibrate(fm, cap)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                if res.multi_face_landmarks:
                    no_person_since = None
                    lm = res.multi_face_landmarks[0].landmark

                    EAR_L, EAR_R, L, R = get_eye_features(lm, *frame.shape[1::-1])
                    eyes_open = EAR_L > EAR_OPEN_THRESH and EAR_R > EAR_OPEN_THRESH

                    eye = (L + R) / 2
                    yaw, pitch = get_head_offset(lm, *frame.shape[1::-1])

                    eye_ok = np.linalg.norm(eye - base_eye) < EYE_REL_THRESH
                    head_ok = abs(yaw - base_yaw) < HEAD_YAW_THRESH and abs(pitch - base_pitch) < HEAD_PITCH_THRESH

                    good = eyes_open and eye_ok and head_ok

                    now = time.time()
                    focus_history.append((now, good))
                    while focus_history and focus_history[0][0] < now - FOCUS_WINDOW_SECONDS:
                        focus_history.popleft()

                    bad_ratio = sum(not x for _, x in focus_history) / max(1, len(focus_history))
                    if bad_ratio >= NOT_FOCUS_THRESHOLD:
                        start_warning_audio()
                    else:
                        stop_warning_audio()

                else:
                    stop_warning_audio()
                    now = time.time()
                    if no_person_since is None:
                        no_person_since = now
                    elif now - no_person_since >= NO_FACE_PROMPT_SEC:
                        speech.speak(
                            "no_person",
                            "Cannot see a person, please adjust the camera",
                            5.0
                        )

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass

    stop_warning_audio()c
    speech.stop()
    cap.release()


if __name__ == "__main__":
    main()