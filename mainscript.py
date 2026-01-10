import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import os
import sys
import subprocess
import threading
import queue

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
# Calibration + thresholds
# -------------------------
CALIB_SECONDS = 2.0        # 校准采样时长（秒）
CALIB_MIN_SAMPLES = 20     # 最少有效采样数

EAR_OPEN_THRESH = 0.20     # 睁眼阈值
EYE_REL_THRESH = 0.12      # 眼睛相对 baseline 偏移阈值
HEAD_YAW_THRESH = 0.15     # 头部 yaw 相对 baseline 阈值
HEAD_PITCH_THRESH = 0.15   # 头部 pitch 相对 baseline 阈值

# 人脸距离/居中判定（基于脸的 bounding box）
FACE_TOO_FAR_RATIO = 0.18   # 脸在画面中的最小占比（宽或高小于该比例则认为太远）
FACE_CENTER_THRESH = 0.18  # 脸中心偏离画面中心的阈值（归一化到 [0,1]）
NO_FACE_PROMPT_SEC = 2.0   # 连续多久检测不到脸才提示

# Linux USB audio device (check with `aplay -l`)
AUDIO_DEVICE_LINUX = "plughw:1,0"

# 滑动窗口参数
FOCUS_WINDOW_SECONDS = 10.0     # 滑动窗口长度（秒）
NOT_FOCUS_THRESHOLD = 0.8       # 在窗口内「不专注」帧的比例阈值

# ---- 语音告警控制（warning.wav） ----
warning_active = False                         # 当前是否处于“正在播警告”的状态
warning_thread = None                          # 播放线程
warning_stop_event = threading.Event()         # 用来通知线程停止
warning_process = None                       # 当前正在播放的子进程

# ---- TTS 提示（校准/状态播报）----
class SpeechManager:
    def __init__(self):
        self.enabled = pyttsx3 is not None
        self._q = queue.Queue()
        self._last_spoken = {}  # key -> timestamp
        self._thread = None
        self._stop = threading.Event()

        if self.enabled:
            try:
                self._engine = pyttsx3.init()
                # 可按需调参
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
                # 一条一条播，避免重叠
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                # 若 TTS 崩了，直接降级为关闭
                self.enabled = False

    def speak(self, key: str, text: str, cooldown_s: float = 3.0):
        """按 key 做冷却，避免同一句反复刷屏"""
        if not self.enabled:
            return
        now = time.time()
        last = self._last_spoken.get(key, 0.0)
        if now - last < cooldown_s:
            return
        self._last_spoken[key] = now
        try:
            # 清掉队列里积压的旧提示（可选，保证更“实时”）
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        self._q.put(text)

    def stop(self):
        self._stop.set()

speech = SpeechManager()


def _get_warning_audio_path() -> str | None:
    """在脚本同目录下寻找 warning.(mp3|wav|aiff)，找到就返回路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ["warning.mp3", "warning.wav", "warning.aiff"]

    for name in candidates:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p

    return None


def _play_warning_once():
    """播放一次 warning 音频，并在收到 stop 事件时立即中断"""
    global warning_process

    audio_path = _get_warning_audio_path()
    if not audio_path or not os.path.exists(audio_path):
        time.sleep(1.0)
        return

    # 启动子进程播放（非阻塞）
    try:
        if sys.platform == "darwin":
            warning_process = subprocess.Popen(["afplay", audio_path])
        elif sys.platform.startswith("linux"):
            # Orange Pi / Linux: use ALSA aplay (ffplay usually not installed)
            warning_process = subprocess.Popen(
                ["aplay", "-D", AUDIO_DEVICE_LINUX, audio_path]
            )
        elif sys.platform.startswith("win"):
            warning_process = subprocess.Popen([
                "powershell", "-Command",
                f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync();'
            ])
        else:
            # 兜底：再试一次 afplay
            warning_process = subprocess.Popen(["afplay", audio_path])
    except Exception as e:
        warning_process = None
        return

    # 轮询子进程状态，一旦收到停止事件就终止
    while warning_process is not None and warning_process.poll() is None:
        if warning_stop_event.is_set():
            try:
                warning_process.terminate()
            except Exception:
                pass
            break
        time.sleep(0.05)

    # 确保子进程结束
    if warning_process is not None and warning_process.poll() is None:
        try:
            warning_process.terminate()
        except Exception:
            pass
    warning_process = None


def _warning_audio_worker():
    """在后台线程中循环播放 warning.wav，直到收到 stop 事件"""
    while not warning_stop_event.is_set():
        _play_warning_once()
        # 可以加一点点 sleep 防止极端情况下忙等
        # time.sleep(0.1)


def start_warning_audio():
    """开始播放语音警告（如果已经在播就不重复启动）"""
    global warning_active, warning_thread

    if warning_active:
        return  # 已经在播了

    warning_active = True
    warning_stop_event.clear()

    warning_thread = threading.Thread(
        target=_warning_audio_worker,
        daemon=True
    )
    warning_thread.start()


def stop_warning_audio():
    """停止播放语音警告（尽可能立即中断当前音频）"""
    global warning_active, warning_thread, warning_process

    if not warning_active:
        return  # 本来就没在播

    warning_active = False
    warning_stop_event.set()  # 通知线程和 _play_warning_once 中的轮询

    # 如果当前子进程还在播，尝试直接终止
    if warning_process is not None and warning_process.poll() is None:
        try:
            warning_process.terminate()
        except Exception:
            pass


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
    """返回 (cx_norm, cy_norm, size_ratio)
    - cx_norm/cy_norm: 脸中心在画面中的归一化坐标 [0,1]
    - size_ratio: min(face_width_ratio, face_height_ratio)
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
    """Orange Pi 上 /dev/video0 可能是 cedrus（不是摄像头）；优先尝试 USB 摄像头 index 1/2。"""
    for idx in (1, 2, 0, 3, 4, 5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
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
            # 校准阶段没检测到脸：提示把摄像头对准
            speech.speak("calib_no_face", "please move camera closer to the center", cooldown_s=2.5)
            continue

        lm = results.multi_face_landmarks[0].landmark

        cx, cy, size_ratio = get_face_box_stats(lm, w, h)
        if size_ratio < FACE_TOO_FAR_RATIO:
            speech.speak("calib_too_far", "Too far away", cooldown_s=2.5)
        # 不强制居中，但偏离很大时也提醒
        if abs(cx - 0.5) > FACE_CENTER_THRESH or abs(cy - 0.5) > FACE_CENTER_THRESH:
            speech.speak("calib_center", "please move camera closer to the center", cooldown_s=2.5)

        EAR_L, EAR_R, Loff, Roff = get_eye_features(lm, w, h)
        eyes_open = (EAR_L > EAR_OPEN_THRESH) and (EAR_R > EAR_OPEN_THRESH)
        if not eyes_open:
            speech.speak("calib_open_eyes", "Cannot detect eyes, please open it", cooldown_s=2.0)
            continue

        eye_offset = (Loff + Roff) / 2.0
        yaw, pitch = get_head_offset(lm, w, h)

        eye_samples.append(eye_offset)
        yaw_samples.append(yaw)
        pitch_samples.append(pitch)
        time.sleep(0.01)

    if len(eye_samples) < CALIB_MIN_SAMPLES:
        # 样本不足时使用 0 baseline（仍可运行，但效果可能差一些）
        speech.speak("calib_failed", "Calibration failed", cooldown_s=0.0)
        return np.array([0.0, 0.0]), 0.0, 0.0

    eye_samples = np.array(eye_samples)
    baseline_eye = np.median(eye_samples, axis=0)
    baseline_yaw = float(np.median(np.array(yaw_samples)))
    baseline_pitch = float(np.median(np.array(pitch_samples)))
    speech.speak("calib_done", "Calibration completed", cooldown_s=0.0)
    return baseline_eye, baseline_yaw, baseline_pitch


def main():
    cap = open_camera_prefer_usb()

    if cap is None:
        raise SystemExit("Camera cannot be opened.")

    # 降低分辨率 → 稳定提升 40%
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # 记录最近一段时间的专注状态（good_focus=True/False）
    focus_history = deque()
    last_face_seen = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:

        # 初始校准：记录当前姿态作为 baseline（不要求直视摄像头）
        stop_warning_audio()
        baseline_eye, baseline_yaw, baseline_pitch = calibrate_baseline(fm, cap)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read error")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                last_face_seen = time.time()
                lm = results.multi_face_landmarks[0].landmark

                # 基于脸 bounding box 的“距离/居中”提示
                cx, cy, size_ratio = get_face_box_stats(lm, w, h)
                if size_ratio < FACE_TOO_FAR_RATIO:
                    speech.speak("too_far", "Too far away", cooldown_s=3.0)
                if abs(cx - 0.5) > FACE_CENTER_THRESH or abs(cy - 0.5) > FACE_CENTER_THRESH:
                    speech.speak("move_center", "please move camera closer to the center", cooldown_s=3.0)

                EAR_L, EAR_R, Loff, Roff = get_eye_features(lm, w, h)
                eyes_open = (EAR_L > EAR_OPEN_THRESH) and (EAR_R > EAR_OPEN_THRESH)
                if not eyes_open:
                    speech.speak("open_eyes", "Cannot detect eyes, please open it", cooldown_s=2.5)

                # 下面保持你原来的逻辑（eye/head focus + 滑窗告警）
                eye_offset = (Loff + Roff) / 2.0
                yaw, pitch = get_head_offset(lm, w, h)

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
                    bad = sum(1 for _, ok in focus_history if not ok)
                    ratio_bad = bad / total

                    if ratio_bad >= NOT_FOCUS_THRESHOLD:
                        start_warning_audio()
                    else:
                        stop_warning_audio()
            else:
                # 连续一段时间没有检测到脸才提示（避免偶尔丢帧就说话）
                if time.time() - last_face_seen > NO_FACE_PROMPT_SEC:
                    speech.speak("no_face", "please move camera closer to the center", cooldown_s=3.0)

            time.sleep(0.05)  # 给 Pi 降负载（FPS ~15）

    cap.release()
    # 退出前保证停止告警
    stop_warning_audio()
    try:
        speech.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()