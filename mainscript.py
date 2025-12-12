import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import os
import sys
import subprocess
import threading

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

# 滑动窗口参数
FOCUS_WINDOW_SECONDS = 10.0     # 滑动窗口长度（秒）
NOT_FOCUS_THRESHOLD = 0.8       # 在窗口内「不专注」帧的比例阈值

# ---- 语音告警控制（warning.wav） ----
warning_active = False                         # 当前是否处于“正在播警告”的状态
warning_thread = None                          # 播放线程
warning_stop_event = threading.Event()         # 用来通知线程停止
warning_process = None                       # 当前正在播放的子进程


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
        print(f"[AUDIO] warning audio not found")
        time.sleep(1.0)
        return

    # 启动子进程播放（非阻塞）
    try:
        if sys.platform == "darwin":
            warning_process = subprocess.Popen(["afplay", audio_path])
        elif sys.platform.startswith("linux"):
            warning_process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
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
        print(f"[AUDIO] Cannot start warning audio: {e}")
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
    print("[AUDIO] START warning.wav")  # 调试用，之后可以删掉


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

    print("[AUDIO] STOP warning.wav")  # 调试用，之后可以删掉


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


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("Camera cannot be opened.")

    # 降低分辨率 → 稳定提升 40%
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # 记录最近一段时间的专注状态（good_focus=True/False）
    focus_history = deque()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read error")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                EAR_L, EAR_R, Loff, Roff = get_eye_features(lm, w, h)

                # 眼睛是否睁开
                eyes_open = (EAR_L > 0.20) and (EAR_R > 0.20)

                # 眼睛注视偏移（左右眼偏移的平均值，归一化在大约 [-0.5, 0.5]）
                eye_offset = (Loff + Roff) / 2.0
                eye_focus = np.linalg.norm(eye_offset) < 0.12  # 越小越严格

                # 头部相对摄像头的偏移（yaw 左右, pitch 上下），归一化在大约 [-1, 1]
                yaw, pitch = get_head_offset(lm, w, h)
                head_center = (abs(yaw) < 0.15) and (abs(pitch) < 0.15)

                # 定义「专注」：眼睛睁开 + 眼睛对准 + 头部对准
                good_focus = eyes_open and eye_focus and head_center

                now = time.time()
                # 记录当前帧的专注状态
                focus_history.append((now, good_focus))

                # 滑动窗口：只保留最近 FOCUS_WINDOW_SECONDS 内的记录
                while focus_history and focus_history[0][0] < now - FOCUS_WINDOW_SECONDS:
                    focus_history.popleft()

                total = len(focus_history)
                if total > 0:
                    bad = sum(1 for _, ok in focus_history if not ok)
                    ratio_bad = bad / total

                    # 如果在窗口内有超过一定比例的「不专注」帧，则触发语音警告
                    if ratio_bad >= NOT_FOCUS_THRESHOLD:
                        start_warning_audio()
                    else:
                        # 恢复专注：停止语音警告
                        stop_warning_audio()

            time.sleep(0.05)  # 给 Pi 降负载（FPS ~15）

    cap.release()
    # 退出前保证停止告警
    stop_warning_audio()


if __name__ == "__main__":
    main()