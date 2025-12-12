import cv2
import numpy as np
import mediapipe as mp
import time

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

# Debug visualization flag and overlay function
DEBUG_VIS = True  # 将可视化集中在一个模块里，测试结束后可以整体删掉或改为 False

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
    cv2.putText(frame, f"EyeOffset: ({eye_offset[0]:.3f}, {eye_offset[1]:.3f})", (10, y0 + 2 * dy_text),
                font, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"HeadCenter: {head_center}", (10, y0 + 3 * dy_text), font, 0.5,
                (0, 255, 0) if head_center else (0, 0, 255), 1)
    cv2.putText(frame, f"HeadOffset yaw: {yaw:.3f}", (10, y0 + 4 * dy_text), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"HeadOffset pitch: {pitch:.3f}", (10, y0 + 5 * dy_text), font, 0.5, (255, 255, 255), 1)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

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

                print(
                    f"EyesOpen={eyes_open}, "
                    f"EyeFocus={eye_focus}, EyeOffset=({eye_offset[0]:.3f}, {eye_offset[1]:.3f}), "
                    f"HeadCenter={head_center}, HeadOffset=(yaw={yaw:.3f}, pitch={pitch:.3f})"
                )

                if DEBUG_VIS:
                    draw_debug_overlay(
                        frame,
                        lm,
                        w,
                        h,
                        eyes_open,
                        eye_focus,
                        eye_offset,
                        yaw,
                        pitch,
                        head_center,
                    )
                    cv2.imshow("debug", frame)
                    # 按 ESC 退出
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            time.sleep(0.05)  # 给 Pi 降负载（FPS ~15）

    if DEBUG_VIS:
        cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
