import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import subprocess
import threading
import queue
import traceback
from collections import deque

try:
    import pyttsx3  # offline TTS (Windows/macOS/Linux)
except Exception:
    pyttsx3 = None

mp_face_mesh = mp.solutions.face_mesh

# 眼睛关键点（MediaPipe FaceMesh 索引）
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362

IRIS_POINTS = list(range(468, 478))  # 虹膜 10 个点

# 头部关键点（大致的脸轮廓与鼻尖，用于估算头部朝向）
FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152
NOSE_TIP = 1

# -------------------------
# Debug / Calibration config
# -------------------------
DEBUG_VIS = True

# Calibration
CALIB_SECONDS = 5.0
CALIB_MIN_SAMPLES = 20

# Thresholds (relative to baseline)
EAR_OPEN_THRESH = 0.20
EYE_REL_THRESH = 0.12
HEAD_YAW_THRESH = 0.15
HEAD_PITCH_THRESH = 0.15

# Face distance/centering prompts (based on face bounding box)
FACE_TOO_FAR_RATIO = 0.18   # if face bbox min(w,h) < this ratio -> too far
FACE_CENTER_THRESH = 0.18   # if |cx-0.5| or |cy-0.5| > this -> ask to center
NO_FACE_PROMPT_SEC = 2.0    # speak if no face detected continuously for this long

# Sliding window warning policy
FOCUS_WINDOW_SECONDS = 10.0
NOT_FOCUS_THRESHOLD = 0.8

# Audio config (Linux USB device)
AUDIO_DEVICE_LINUX = "plughw:1,0"  # change if your USB audio card differs (check: aplay -l)
WARNING_AUDIO_CANDIDATES = ["warning.wav", "warning.aiff", "warning.mp3"]

# -------------------------
# Audio warning (loop until focused)
# -------------------------
warning_active = False
warning_thread = None
warning_stop_event = threading.Event()
warning_process = None

# -------------------------
# TTS prompts (calibration / status)
# -------------------------
class SpeechManager:
    def __init__(self):
        self.enabled = pyttsx3 is not None
        self._q = queue.Queue()
        self._last_spoken = {}  # key -> timestamp
        self._stop = threading.Event()
        self._thread = None

        if self.enabled:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', 180)
                self._engine.setProperty('volume', 1.0)
            except Exception:
                self.enabled = False
                self._engine = None

        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

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

    def speak(self, key: str, text: str, cooldown_s: float = 3.0):
        """Speak with per-key cooldown to avoid spamming."""
        if not self.enabled:
            return
        now = time.time()
        last = self._last_spoken.get(key, 0.0)
        if now - last < cooldown_s:
            return
        self._last_spoken[key] = now

        # keep newest prompt (avoid a queue backlog)
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass

        self._q.put(text)

    def stop(self):
        self._stop.set()


speech = SpeechManager()

# -------------------------
# UI helpers
# -------------------------
def _window_closed(win_name: str) -> bool:
    """True if OpenCV window is closed/hidden."""
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return False


def _get_warning_audio_path() -> str | None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in WARNING_AUDIO_CANDIDATES:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    return None


def _play_warning_once_interruptible():
    """Play one warning clip; terminate immediately if stop requested."""
    global warning_process

    audio_path = _get_warning_audio_path()
    if not audio_path:
        time.sleep(1.0)
        return

    try:
        if sys.platform == "darwin":
            warning_process = subprocess.Popen(["afplay", audio_path])
        elif sys.platform.startswith("linux"):
            # Orange Pi / Linux: use ALSA aplay
            warning_process = subprocess.Popen(["aplay", "-D", AUDIO_DEVICE_LINUX, audio_path])
        elif sys.platform.startswith("win"):
            warning_process = subprocess.Popen([
                "powershell", "-Command",
                f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync();'
            ])
        else:
            warning_process = subprocess.Popen(["afplay", audio_path])
    except FileNotFoundError as e:
        print(f"[AUDIO] player not found: {e}")
        warning_process = None
        warning_stop_event.set()
        return
    except Exception as e:
        print(f"[AUDIO] Cannot start warning audio: {e}")
        warning_process = None
        return

    while warning_process is not None and warning_process.poll() is None:
        if warning_stop_event.is_set():
            try:
                warning_process.terminate()
            except Exception:
                pass
            break
        time.sleep(0.03)

    if warning_process is not None and warning_process.poll() is None:
        try:
            warning_process.terminate()
        except Exception:
            pass
    warning_process = None


def _warning_audio_worker():
    while not warning_stop_event.is_set():
        _play_warning_once_interruptible()


def start_warning_audio():
    global warning_active, warning_thread
    if warning_active:
        return
    warning_active = True
    warning_stop_event.clear()
    warning_thread = threading.Thread(target=_warning_audio_worker, daemon=True)
    warning_thread.start()


def stop_warning_audio():
    global warning_active, warning_process
    if not warning_active:
        return
    warning_active = False
    warning_stop_event.set()
    if warning_process is not None and warning_process.poll() is None:
        try:
            warning_process.terminate()
        except Exception:
            pass


def draw_debug_overlay(frame,
                       landmarks,
                       w,
                       h,
                       eyes_open,
                       eye_focus,
                       eye_offset,
                       yaw,
                       pitch,
                       head_center):
    """调试用可视化：在画面上画出眼睛/虹膜/头部朝向以及文字信息"""
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]

    # 眼睛关键点像素坐标
    L_top = pts[LEFT_EYE_TOP]
    L_bottom = pts[LEFT_EYE_BOTTOM]
    L_left = pts[LEFT_EYE_LEFT]
    L_right = pts[LEFT_EYE_RIGHT]

    R_top = pts[RIGHT_EYE_TOP]
    R_bottom = pts[RIGHT_EYE_BOTTOM]
    R_left = pts[RIGHT_EYE_LEFT]
    R_right = pts[RIGHT_EYE_RIGHT]

    # 虹膜像素点
    iris_pts = np.array([pts[i] for i in IRIS_POINTS])
    median_x = np.median(iris_pts[:, 0])
    left_iris_pts = iris_pts[iris_pts[:, 0] < median_x]
    right_iris_pts = iris_pts[iris_pts[:, 0] >= median_x]

    if len(left_iris_pts) > 0:
        left_iris_center = left_iris_pts.mean(axis=0).astype(int)
    else:
        left_iris_center = None

    if len(right_iris_pts) > 0:
        right_iris_center = right_iris_pts.mean(axis=0).astype(int)
    else:
        right_iris_center = None

    # 脸关键点
    face_left = pts[FACE_LEFT]
    face_right = pts[FACE_RIGHT]
    face_top = pts[FACE_TOP]
    face_bottom = pts[FACE_BOTTOM]
    nose = pts[NOSE_TIP]

    # 画出眼睛轮廓
    eye_color = (0, 255, 0) if eye_focus else (0, 255, 255)
    cv2.line(frame, L_left, L_right, eye_color, 1)
    cv2.line(frame, L_top, L_bottom, eye_color, 1)
    cv2.line(frame, R_left, R_right, eye_color, 1)
    cv2.line(frame, R_top, R_bottom, eye_color, 1)

    # 画虹膜中心
    if left_iris_center is not None:
        cv2.circle(frame, tuple(left_iris_center), 2, (255, 0, 0), -1)
    if right_iris_center is not None:
        cv2.circle(frame, tuple(right_iris_center), 2, (255, 0, 0), -1)

    # 画脸的大致边框和鼻尖
    cv2.circle(frame, face_left, 2, (0, 0, 255), -1)
    cv2.circle(frame, face_right, 2, (0, 0, 255), -1)
    cv2.circle(frame, face_top, 2, (0, 0, 255), -1)
    cv2.circle(frame, face_bottom, 2, (0, 0, 255), -1)
    cv2.circle(frame, nose, 3, (0, 0, 255), -1)

    # 用一个小箭头显示头部 yaw/pitch（鼻尖指向）
    arrow_scale = 40  # 可以调节大小
    dx = int(yaw * arrow_scale)
    dy = int(pitch * arrow_scale)
    cv2.arrowedLine(frame, nose, (nose[0] + dx, nose[1] + dy), (0, 0, 255), 2, tipLength=0.3)

    # 文字信息
    y0 = 20
    dy_text = 18
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"EyesOpen: {eyes_open}", (10, y0), font, 0.5, (0, 255, 0) if eyes_open else (0, 0, 255), 1)
    cv2.putText(frame, f"EyeFocus: {eye_focus}", (10, y0 + dy_text), font, 0.5,
                (0, 255, 0) if eye_focus else (0, 255, 255), 1)
    cv2.putText(frame, f"EyeRel: ({eye_offset[0]:.3f}, {eye_offset[1]:.3f})", (10, y0 + 2 * dy_text),
                font, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"HeadCenter: {head_center}", (10, y0 + 3 * dy_text), font, 0.5,
                (0, 255, 0) if head_center else (0, 0, 255), 1)
    cv2.putText(frame, f"HeadRel yaw: {yaw:.3f}", (10, y0 + 4 * dy_text), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"HeadRel pitch: {pitch:.3f}", (10, y0 + 5 * dy_text), font, 0.5, (255, 255, 255), 1)

def EAR(top, bottom, left, right):
    """眼睛纵横比，用于判断睁眼/闭眼"""
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / (horizontal + 1e-6)

def get_eye_features(landmarks, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]

    # 左眼关键点
    L_top = np.array(pts[LEFT_EYE_TOP])
    L_bottom = np.array(pts[LEFT_EYE_BOTTOM])
    L_left = np.array(pts[LEFT_EYE_LEFT])
    L_right = np.array(pts[LEFT_EYE_RIGHT])

    # 右眼关键点
    R_top = np.array(pts[RIGHT_EYE_TOP])
    R_bottom = np.array(pts[RIGHT_EYE_BOTTOM])
    R_left = np.array(pts[RIGHT_EYE_LEFT])
    R_right = np.array(pts[RIGHT_EYE_RIGHT])

    # 眼睛中心（用于判断是否直视）
    L_center = (L_left + L_right) / 2
    R_center = (R_left + R_right) / 2

    # 虹膜中心点
    iris_pts = np.array([pts[i] for i in IRIS_POINTS])
    median_x = np.median(iris_pts[:, 0])
    left_iris = iris_pts[iris_pts[:, 0] < median_x].mean(axis=0)
    right_iris = iris_pts[iris_pts[:, 0] >= median_x].mean(axis=0)

    # EAR 计算睁眼/闭眼
    EAR_left = EAR(L_top, L_bottom, L_left, L_right)
    EAR_right = EAR(R_top, R_bottom, R_left, R_right)

    # 虹膜中心相对于眼睛中心的偏移量（判断直视）
    L_offset = (left_iris - L_center) / np.linalg.norm(L_left - L_right)
    R_offset = (right_iris - R_center) / np.linalg.norm(R_left - R_right)



    return EAR_left, EAR_right, L_offset, R_offset

def get_head_offset(landmarks, w, h):
    """粗略估算头部相对摄像头的偏转（yaw 左右，pitch 上下），单位为归一化偏移"""
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]

    face_left = np.array(pts[FACE_LEFT])
    face_right = np.array(pts[FACE_RIGHT])
    face_top = np.array(pts[FACE_TOP])
    face_bottom = np.array(pts[FACE_BOTTOM])
    nose = np.array(pts[NOSE_TIP])

    # 脸的大致中心
    face_center = (face_left + face_right + face_top + face_bottom) / 4.0

    # 使用脸宽和脸高做归一化，让 yaw/pitch 大致落在 [-1, 1]
    face_width = np.linalg.norm(face_right - face_left) + 1e-6
    face_height = np.linalg.norm(face_bottom - face_top) + 1e-6

    yaw = (nose[0] - face_center[0]) / face_width      # 左右偏转
    pitch = (nose[1] - face_center[1]) / face_height   # 上下偏转

    return float(yaw), float(pitch)


def get_face_box_stats(landmarks, w, h):
    """Return (cx_norm, cy_norm, size_ratio)
    cx_norm/cy_norm in [0,1], size_ratio = min(face_w, face_h) in normalized coords.
    """
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    face_w = maxx - minx
    face_h = maxy - miny
    size_ratio = float(min(face_w, face_h))
    return float(cx), float(cy), size_ratio


def open_camera_prefer_usb():
    """Orange Pi 上 /dev/video0 可能是 cedrus；优先尝试 1/2。"""
    for idx in (0, 1, 2, 0, 3, 4, 5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Using camera index {idx}")
            return cap
        cap.release()
    return None


def calibrate_baseline(fm, cap):
    """采集 CALIB_SECONDS 秒样本，取 median 作为 baseline（不要求看镜头）。"""
    speech.speak("calib_start", "Calibration starting, please don't move your head", cooldown_s=0.0)
    eye_samples = []
    yaw_samples = []
    pitch_samples = []

    start_t = time.time()
    while time.time() - start_t < CALIB_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)
        if not results.multi_face_landmarks:
            continue

        lm = results.multi_face_landmarks[0].landmark
        cx, cy, size_ratio = get_face_box_stats(lm, w, h)

        EAR_L, EAR_R, Loff, Roff = get_eye_features(lm, w, h)
        eyes_open = (EAR_L > EAR_OPEN_THRESH) and (EAR_R > EAR_OPEN_THRESH)
        if not eyes_open:
            continue

        eye_offset = (Loff + Roff) / 2.0
        yaw, pitch = get_head_offset(lm, w, h)

        eye_samples.append(eye_offset)
        yaw_samples.append(yaw)
        pitch_samples.append(pitch)
        time.sleep(0.01)

    if len(eye_samples) < CALIB_MIN_SAMPLES:
        print(f"[CALIB] Not enough samples ({len(eye_samples)}). Using zero baseline.")
        speech.speak("calib_failed", "Calibration failed", cooldown_s=0.0)
        return np.array([0.0, 0.0]), 0.0, 0.0

    eye_samples = np.array(eye_samples)
    baseline_eye = np.median(eye_samples, axis=0)
    baseline_yaw = float(np.median(np.array(yaw_samples)))
    baseline_pitch = float(np.median(np.array(pitch_samples)))

    print(
        f"[CALIB] Done. baseline_eye=({baseline_eye[0]:.3f},{baseline_eye[1]:.3f}) "
        f"baseline_yaw={baseline_yaw:.3f} baseline_pitch={baseline_pitch:.3f}"
    )
    speech.speak("calib_done", "Calibration completed", cooldown_s=0.0)
    return baseline_eye, baseline_yaw, baseline_pitch


def main():
    cap = open_camera_prefer_usb()
    if cap is None:
        raise SystemExit("Camera cannot be opened.")

    # 降低分辨率 → 稳定提升 40%
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:

        # initial calibration
        stop_warning_audio()
        baseline_eye, baseline_yaw, baseline_pitch = calibrate_baseline(fm, cap)

        focus_history = deque()
        no_person_since = None
        cam_fail_streak = 0

        while True:
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cam_fail_streak += 1
                    if cam_fail_streak == 1 or cam_fail_streak % 30 == 0:
                        print(f"[WARN] Camera read failed (streak={cam_fail_streak}). Retrying...")
                    time.sleep(0.05)
                    continue
                cam_fail_streak = 0

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = fm.process(rgb)

                # Defaults so DEBUG_VIS can always render
                lm = None
                eyes_open = False
                eye_focus = False
                head_center = False
                eye_rel = np.array([0.0, 0.0])
                yaw_rel = 0.0
                pitch_rel = 0.0

                if results.multi_face_landmarks:
                    no_person_since = None
                    lm = results.multi_face_landmarks[0].landmark

                    EAR_L, EAR_R, Loff, Roff = get_eye_features(lm, w, h)
                    eyes_open = (EAR_L > EAR_OPEN_THRESH) and (EAR_R > EAR_OPEN_THRESH)
                    if not eyes_open:
                        pass

                    eye_offset = (Loff + Roff) / 2.0
                    yaw, pitch = get_head_offset(lm, w, h)

                    # relative to baseline
                    eye_rel = eye_offset - baseline_eye
                    yaw_rel = yaw - baseline_yaw
                    pitch_rel = pitch - baseline_pitch

                    eye_focus = np.linalg.norm(eye_rel) < EYE_REL_THRESH
                    head_center = (abs(yaw_rel) < HEAD_YAW_THRESH) and (abs(pitch_rel) < HEAD_PITCH_THRESH)

                    good_focus = eyes_open and eye_focus and head_center

                    now = time.time()
                    focus_history.append((now, good_focus))
                    while focus_history and focus_history[0][0] < now - FOCUS_WINDOW_SECONDS:
                        focus_history.popleft()

                    total = len(focus_history)
                    if total > 0:
                        bad = sum(1 for _, okk in focus_history if not okk)
                        ratio_bad = bad / total
                        if ratio_bad >= NOT_FOCUS_THRESHOLD:
                            start_warning_audio()
                        else:
                            stop_warning_audio()
                else:
                    # No face detected:
                    # 1) 不更新 focus_history
                    # 2) 不计算 ratio_bad
                    # 3) 强制停止 warning.wav
                    stop_warning_audio()

                    now = time.time()
                    if no_person_since is None:
                        no_person_since = now
                    elif now - no_person_since >= NO_FACE_PROMPT_SEC:
                        speech.speak(
                            "no_person",
                            "Cannot see a person, please adjust the camera",
                            cooldown_s=5.0
                        )

                # Always handle UI + keys when DEBUG_VIS is enabled
                if DEBUG_VIS:
                    if lm is not None:
                        draw_debug_overlay(
                            frame,
                            lm,
                            w,
                            h,
                            eyes_open,
                            eye_focus,
                            eye_rel,
                            yaw_rel,
                            pitch_rel,
                            head_center,
                        )
                    else:
                        cv2.putText(frame, "No face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    cv2.putText(frame, "Keys: c=recalibrate, q=quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow("debug", frame)

                    if _window_closed("debug"):
                        print("[INFO] Debug window closed. Exiting.")
                        break

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        print("[INFO] Quit key pressed. Exiting.")
                        break
                    if k == ord('c'):
                        print("[INFO] Recalibrate requested.")
                        stop_warning_audio()
                        focus_history.clear()
                        baseline_eye, baseline_yaw, baseline_pitch = calibrate_baseline(fm, cap)

                time.sleep(0.05)  # 给 Pi 降负载（FPS ~15）

            except Exception as e:
                print(f"[ERROR] Exception in main loop: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                continue

    try:
        stop_warning_audio()
    except Exception:
        pass
    try:
        speech.stop()
    except Exception:
        pass
    if DEBUG_VIS:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    try:
        cap.release()
    except Exception:
        pass

if __name__ == "__main__":
    main()
