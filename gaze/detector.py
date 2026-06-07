"""Platform-agnostic focus detection core.

Wraps MediaPipe FaceMesh and reproduces the calibration + per-frame
classification + rolling-window focus logic from the original main.py,
but decoupled from audio/output so it can drive any front-end.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp

from . import config as C

log = logging.getLogger("gaze")

mp_face_mesh = mp.solutions.face_mesh


class FocusState(Enum):
    FOCUSED = "focused"
    DROWSY = "drowsy"
    LOOKING_AWAY = "looking_away"
    HEAD_TURNED = "head_turned"
    NO_FACE = "no_face"


# -------- Math helpers (ported verbatim) --------
def _ear(top, bottom, left, right):
    return np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)


def get_eye_features(lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    L_top = np.array(pts[C.LEFT_EYE_TOP])
    L_bottom = np.array(pts[C.LEFT_EYE_BOTTOM])
    L_left = np.array(pts[C.LEFT_EYE_LEFT])
    L_right = np.array(pts[C.LEFT_EYE_RIGHT])
    R_top = np.array(pts[C.RIGHT_EYE_TOP])
    R_bottom = np.array(pts[C.RIGHT_EYE_BOTTOM])
    R_left = np.array(pts[C.RIGHT_EYE_LEFT])
    R_right = np.array(pts[C.RIGHT_EYE_RIGHT])

    iris = np.array([pts[i] for i in C.IRIS_POINTS])
    midx = np.median(iris[:, 0])
    L_iris = iris[iris[:, 0] < midx].mean(axis=0)
    R_iris = iris[iris[:, 0] >= midx].mean(axis=0)

    EAR_L = _ear(L_top, L_bottom, L_left, L_right)
    EAR_R = _ear(R_top, R_bottom, R_left, R_right)

    L_off = (L_iris - (L_left + L_right) / 2) / np.linalg.norm(L_left - L_right)
    R_off = (R_iris - (R_left + R_right) / 2) / np.linalg.norm(R_left - R_right)

    return EAR_L, EAR_R, L_off, R_off


def get_head_offset(lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    face_c = (np.array(pts[C.FACE_LEFT]) + np.array(pts[C.FACE_RIGHT]) +
              np.array(pts[C.FACE_TOP]) + np.array(pts[C.FACE_BOTTOM])) / 4
    nose = np.array(pts[C.NOSE_TIP])
    fw = np.linalg.norm(np.array(pts[C.FACE_RIGHT]) - np.array(pts[C.FACE_LEFT])) + 1e-6
    fh = np.linalg.norm(np.array(pts[C.FACE_BOTTOM]) - np.array(pts[C.FACE_TOP])) + 1e-6
    return (nose[0] - face_c[0]) / fw, (nose[1] - face_c[1]) / fh


def get_face_metrics(lm, w, h):
    """Returns (face_width_ratio, center_x_ratio, center_y_ratio)."""
    pts = [(p.x * w, p.y * h) for p in lm]
    face_w = abs(pts[C.FACE_RIGHT][0] - pts[C.FACE_LEFT][0])
    cx = (pts[C.FACE_LEFT][0] + pts[C.FACE_RIGHT][0]) / 2
    cy = (pts[C.FACE_TOP][1] + pts[C.FACE_BOTTOM][1]) / 2
    return face_w / w, cx / w, cy / h


def get_calib_hints(lm, w, h, EAR_L, EAR_R):
    """Prioritised hints to get the face into a sampleable position."""
    hints = []
    face_w, cx, cy = get_face_metrics(lm, w, h)
    EDGE = 0.04

    if face_w < C.CALIB_FACE_SIZE_MIN:
        hints.append("Try moving the camera a little closer")
    elif cx - face_w / 2 < EDGE:
        hints.append("Part of your face is cut off — shift the camera right")
    elif cx + face_w / 2 > 1.0 - EDGE:
        hints.append("Part of your face is cut off — shift the camera left")
    elif cy - face_w / 2 < EDGE:
        hints.append("Part of your face is cut off — angle the camera down")
    elif cy + face_w / 2 > 1.0 - EDGE:
        hints.append("Part of your face is cut off — angle the camera up")

    if EAR_L < C.EAR_OPEN_THRESH or EAR_R < C.EAR_OPEN_THRESH:
        hints.append("Open your eyes fully so I can see them clearly")

    return hints


def _configure(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_HEIGHT)
    return cap


# ---------- camera detection ----------
# Cache the inventory for the lifetime of the process so we don't shell out on
# every Switch-camera click. The user can flip Continuity on/off between runs.
_INVENTORY_CACHE = None


def _avfoundation_via_pyobjc_detailed():
    """Like _avfoundation_via_pyobjc but returns dicts with `name` and
    `device_type` (e.g. AVCaptureDeviceTypeContinuityCamera). Empty list if
    PyObjC isn't installed."""
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation
        from AVFoundation import AVCaptureDevice
    except Exception:
        return []

    out = []
    try:
        from AVFoundation import AVCaptureDeviceDiscoverySession
        type_names = [
            "AVCaptureDeviceTypeBuiltInWideAngleCamera",
            "AVCaptureDeviceTypeExternal",
            "AVCaptureDeviceTypeContinuityCamera",
            "AVCaptureDeviceTypeDeskViewCamera",
        ]
        types = []
        for n in type_names:
            try:
                types.append(getattr(AVFoundation, n))
            except AttributeError:
                pass
        if types:
            sess = AVCaptureDeviceDiscoverySession.\
                discoverySessionWithDeviceTypes_mediaType_position_(
                    types, "vide", 0)
            for d in (sess.devices() or []):
                out.append({
                    "name": str(d.localizedName()),
                    "device_type": str(d.deviceType()),
                })
            if out:
                return out
    except Exception as e:
        log.debug("AVCaptureDeviceDiscoverySession failed: %s", e)

    try:
        for d in AVCaptureDevice.devicesWithMediaType_("vide"):
            out.append({
                "name": str(d.localizedName()),
                "device_type": "",
            })
        return out
    except Exception:
        return []


def _avfoundation_via_pyobjc():
    """Backwards-compat shim — returns just names."""
    return [d["name"] for d in _avfoundation_via_pyobjc_detailed()]


def _avfoundation_via_ffmpeg():
    """Ground truth via `ffmpeg -f avfoundation -list_devices true -i ""`.
    The output is on stderr and looks like:
        [AVFoundation indev @ 0x…] AVFoundation video devices:
        [AVFoundation indev @ 0x…] [0] FaceTime HD Camera
        [AVFoundation indev @ 0x…] [1] iPhone (5) Camera
        ...
        [AVFoundation indev @ 0x…] AVFoundation audio devices:
    We parse the video section only. ffmpeg's avfoundation indev uses the
    same legacy `devicesWithMediaType:` enumeration as OpenCV's
    cap_avfoundation_mac, so the indices match."""
    if sys.platform != "darwin":
        return []
    if not shutil.which("ffmpeg"):
        return []
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5.0)
    except Exception:
        return []
    text = (proc.stderr or "") + (proc.stdout or "")
    out = []
    in_video = False
    pat = re.compile(r"\[AVFoundation[^\]]*\]\s*\[(\d+)\]\s*(.+?)\s*$")
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
            continue
        if not in_video:
            continue
        m = pat.search(line)
        if not m:
            continue
        idx, name = int(m.group(1)), m.group(2).strip()
        # screen capture devices show up here too; we don't want them
        if name.lower().startswith("capture screen"):
            continue
        while len(out) <= idx:
            out.append("")
        out[idx] = name
    # strip trailing empties (e.g. skipped screen-capture slots at the end)
    while out and not out[-1]:
        out.pop()
    return out


def _system_profiler_inventory():
    """Last-resort fallback. Known to *hide* Continuity Camera entries even
    when AVFoundation is actively exposing them — that's why we prefer the
    AVFoundation-direct paths above."""
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType", "-json"],
            stderr=subprocess.DEVNULL, timeout=3.0)
        data = json.loads(out)
        return [(c.get("_name") or "").strip()
                for c in data.get("SPCameraDataType", [])]
    except Exception:
        return []


def _macos_camera_inventory():
    """Camera list in cv2-VideoCapture index order. Tries PyObjC, then
    ffmpeg, then system_profiler. Cached for the process lifetime."""
    global _INVENTORY_CACHE
    if _INVENTORY_CACHE is not None:
        return _INVENTORY_CACHE
    inv, src = [], "none"
    for fn, label in ((_avfoundation_via_pyobjc, "pyobjc"),
                      (_avfoundation_via_ffmpeg, "ffmpeg"),
                      (_system_profiler_inventory, "system_profiler")):
        inv = fn()
        if inv:
            src = label
            break
    if inv:
        log.info("Camera inventory (%s): %s",
                 src, ", ".join(f"[{i}] {n}" for i, n in enumerate(inv)))
    _INVENTORY_CACHE = inv
    return inv


def refresh_camera_inventory():
    """Invalidate the cached inventory (e.g. after the user attaches/removes
    an iPhone). Safe to call freely; the next probe will re-enumerate."""
    global _INVENTORY_CACHE
    _INVENTORY_CACHE = None


def _is_continuity_camera(name):
    """iPhone/iPad showing up as a webcam via macOS Continuity Camera."""
    n = (name or "").lower()
    return "iphone" in n or "ipad" in n or "continuity" in n


def blocked_camera_indices():
    """Indices that auto-probe should avoid (iPhone/iPad Continuity Camera).
    The user can still target them explicitly via the GAZE_CAMERA_INDEX env var."""
    return {i for i, name in enumerate(_macos_camera_inventory())
            if _is_continuity_camera(name)}


def camera_name(i):
    inv = _macos_camera_inventory()
    return inv[i] if 0 <= i < len(inv) else None


# AVAuthorizationStatus enum values (AVFoundation):
#   0 = NotDetermined, 1 = Restricted, 2 = Denied, 3 = Authorized
_AV_AUTH_NOT_DETERMINED = 0
_AV_AUTH_RESTRICTED = 1
_AV_AUTH_DENIED = 2
_AV_AUTH_AUTHORIZED = 3


def request_camera_permission(timeout=60.0):
    """Synchronously request camera permission on macOS.

    Fix for the first-launch failure: on a fresh install, cv2/ffmpeg open the
    camera while macOS is still showing the TCC permission dialog. The open
    call returns instantly (no frames), the monitor gives up, and only the
    *second* launch — after permission has been granted — works. By calling
    requestAccessForMediaType_completionHandler_ up front, we block the
    monitor thread until the user actually clicks Allow / Deny, so the
    subsequent open call is guaranteed to be made with a settled permission.

    Returns True if permission is granted (or non-macOS, or PyObjC missing);
    False if the user denied / system restricted access.
    """
    if sys.platform != "darwin":
        return True
    try:
        from AVFoundation import AVCaptureDevice
    except Exception:
        log.warning("PyObjC AVFoundation unavailable; skipping permission "
                    "request and hoping for the best.")
        return True

    try:
        status = int(AVCaptureDevice.authorizationStatusForMediaType_("vide"))
    except Exception:
        log.exception("authorizationStatusForMediaType_ failed; "
                      "assuming permission granted.")
        return True

    if status == _AV_AUTH_AUTHORIZED:
        log.info("Camera permission already granted.")
        return True
    if status in (_AV_AUTH_RESTRICTED, _AV_AUTH_DENIED):
        log.error("Camera permission denied/restricted (status=%d). The user "
                  "must enable Gaze Nailong under System Settings → Privacy "
                  "& Security → Camera, then reopen the app.", status)
        return False

    log.info("Camera permission status=notDetermined — prompting user and "
             "blocking the monitor thread until they respond.")

    import threading as _t
    done = _t.Event()
    result = {"granted": False}

    def _handler(granted):
        result["granted"] = bool(granted)
        done.set()

    try:
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            "vide", _handler)
    except Exception:
        log.exception("requestAccessForMediaType_completionHandler_ raised; "
                      "falling back to optimistic 'granted'.")
        return True

    if not done.wait(timeout=timeout):
        log.warning("Camera permission prompt did not return within %.1fs.",
                    timeout)
        return False

    log.info("Camera permission %s by user.",
             "granted" if result["granted"] else "denied")
    return result["granted"]


def _env_forced_camera_index():
    """GAZE_CAMERA_INDEX=N forces a specific cv2 index. Wins over everything,
    including the blocked list — so users can re-enable the iPhone explicitly."""
    raw = os.environ.get("GAZE_CAMERA_INDEX")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("Ignoring invalid GAZE_CAMERA_INDEX=%r (expected an integer).", raw)
        return None


def _fingerprint_camera(cap, label=""):
    """Capture several signals from an open cv2.VideoCapture so we can score
    how iPhone-like it is. Returns a dict with at minimum `default_w`,
    `low_res_w`, `first_frame_ms`, and `frames_ok`."""
    info = {"default_w": 0, "default_h": 0,
            "low_res_w": 0, "low_res_h": 0,
            "first_frame_ms": -1.0, "frames_ok": 0}

    # 1) Default native frame (no SETs yet).
    t0 = time.time()
    ok, frame = False, None
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            break
        time.sleep(0.07)
    info["first_frame_ms"] = (time.time() - t0) * 1000
    if ok and frame is not None and frame.size:
        info["default_w"] = int(frame.shape[1])
        info["default_h"] = int(frame.shape[0])
        info["frames_ok"] += 1

    # 2) Frame size after requesting 320×240.
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    except Exception:
        pass
    for _ in range(12):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            info["low_res_w"] = int(frame.shape[1])
            info["low_res_h"] = int(frame.shape[0])
            info["frames_ok"] += 1
            break
        time.sleep(0.06)

    log.info("Probe %s: default=%dx%d, low=%dx%d, first_frame_ms=%.0f, frames_ok=%d",
             label, info["default_w"], info["default_h"],
             info["low_res_w"], info["low_res_h"],
             info["first_frame_ms"], info["frames_ok"])
    return info


def _score_as_builtin(info, device_type=""):
    """Score how likely a camera is the built-in MacBook camera.

    DESIGN NOTE (post-mortem): I originally tried to use first-frame latency
    as the dominant signal. It backfires on this user's machine because:
      - FaceTime HD on M2 MacBook Air can take 700–1200ms to deliver its
        first frame from a cold open (cold start, no buffers yet).
      - iPhone Continuity Camera, once paired and streaming, often delivers
        its first frame in <50ms because it's already running.
    So `fast = built-in` and `slow = iPhone` are both wrong here. The
    PyObjC device type from `AVCaptureDeviceDiscoverySession` IS authoritative
    when present — it matches cv2's enumeration on macOS 14+.

    Signals (in priority order):
      1. PyObjC device_type — primary, definitive when available
      2. low_res request honored — fallback signal
      3. default 4K capability — fallback (iPhone-specific)
      4. first_frame_ms — small confirmation bonus only
    """
    score = 0
    dt = (device_type or "").lower()

    # 1. Primary: PyObjC device type. When PyObjC tells us, we listen.
    if "builtin" in dt:
        return 200  # short-circuit threshold is 100 — guaranteed to win
    if "continuity" in dt or "deskview" in dt:
        return -200  # never pick the iPhone unless it's the only option

    # When type is missing or "External" (which on macOS 14+ without
    # NSCameraUseContinuityCameraDeviceType in Info.plist can mean iPhone),
    # fall back to behavioral signals.
    if "external" in dt:
        score -= 30  # External is suspicious — could be iPhone in disguise

    w_low = info.get("low_res_w", 0)
    if w_low and w_low <= 640:
        score += 20
    elif w_low and w_low >= 1100:
        score -= 20

    w_def = info.get("default_w", 0)
    if w_def >= 3000:
        score -= 30

    t_ff = info.get("first_frame_ms", -1)
    if 0 < t_ff < 250:
        score += 10
    elif t_ff > 1200:
        score -= 10
    return score


def _is_iphone_by_behavior(cap):
    """Backwards-compat boolean wrapper around the fingerprint score."""
    info = _fingerprint_camera(cap, label="(legacy check)")
    if info["frames_ok"] == 0:
        return None
    return _score_as_builtin(info) < 0


def open_camera(preferred=None):
    """Open a camera, biased hard against macOS Continuity Camera.

    Strategy: probe EVERY working cv2 index, score each by a multi-signal
    fingerprint (PyObjC device type if available + behavioral fingerprint),
    and open the highest-scoring one. Logs every candidate so you can see
    why it picked what it picked. The behavioral test combines four signals
    so one misbehaving signal can't push us onto the iPhone:

      * AVFoundation device type (definitive when PyObjC is installed)
      * Low-res request honored? (built-ins yes, iPhone no)
      * First-frame latency (built-ins ~100ms, iPhone 800ms+)
      * Default resolution ≥ 3000px (only iPhone supports 4K)

    Resolution order:
      1. GAZE_CAMERA_INDEX env var (force, no questions asked).
      2. Best-scoring candidate from full probe.
      3. `preferred` from settings.json, if it could be opened at all.

    Returns (capture, index) or (None, None)."""
    refresh_camera_inventory()
    inv = _macos_camera_inventory()
    detailed = _avfoundation_via_pyobjc_detailed()
    if inv:
        log.info("Camera inventory: %s",
                 ", ".join(f"[{i}] {n}" for i, n in enumerate(inv)))
    if detailed:
        log.info("AVFoundation device types: %s",
                 ", ".join(f"[{i}] {d['device_type']}" for i, d in enumerate(detailed)))

    forced = _env_forced_camera_index()
    if forced is not None:
        cap = cv2.VideoCapture(forced)
        if cap.isOpened():
            log.info("Opened camera index %s (forced via GAZE_CAMERA_INDEX).", forced)
            return _configure(cap), forced
        cap.release()
        log.warning("GAZE_CAMERA_INDEX=%s did not open; falling through.", forced)

    # PLATINUM PATH: ffmpeg-by-name bypass. cv2's AVFoundation enumeration on
    # anaconda opencv-python sometimes omits the BuiltIn camera entirely, so
    # every cv2 index lands on the iPhone Continuity feed. When PyObjC reports
    # a BuiltIn device and ffmpeg is installed, we open it by NAME through
    # ffmpeg's avfoundation indev — this targets the FaceTime HD specifically,
    # regardless of what cv2 enumerates.
    if not os.environ.get("GAZE_DISABLE_FFMPEG_CAPTURE"):
        try:
            from .ffcapture import FFmpegCamera, ffmpeg_available
            if ffmpeg_available() and detailed:
                # Prefer the localizedName of the BuiltIn device.
                builtin_names = [d.get("name") or ""
                                 for d in detailed
                                 if "builtin" in (d.get("device_type") or "").lower()
                                 and d.get("name")]
                # Fallback to "FaceTime HD Camera" generic name if PyObjC didn't
                # name it (unlikely).
                if not builtin_names:
                    builtin_names = ["FaceTime HD Camera"]
                for name in builtin_names:
                    log.info("Trying ffmpeg avfoundation by name: %r", name)
                    cap = FFmpegCamera(name,
                                       width=C.FRAME_WIDTH,
                                       height=C.FRAME_HEIGHT,
                                       fps=30)
                    if cap.isOpened():
                        ok, frame = cap.read()
                        if ok and frame is not None and frame.size:
                            log.info("WINNER (ffmpeg-by-name): %r — cv2 bypassed.", name)
                            # Tag the capture so the monitor can show the
                            # device name in the sidebar even though we don't
                            # have a real cv2 index.
                            cap._gaze_name = name
                            # cv2-index -1 is our sentinel for "ffmpeg-managed".
                            return cap, -1
                        cap.release()
                        log.warning("ffmpeg opened %r but no frame arrived.", name)
                    else:
                        log.warning("ffmpeg could not open %r.", name)
        except Exception:
            log.exception("ffmpeg-by-name capture failed; falling through.")

    # GOLDEN PATH: PyObjC says one of the cv2 indices is the BuiltIn camera.
    # On macOS 14+ the discovery-session enumeration matches cv2's order, so
    # we open ONLY that index and use it — iPhone Continuity entries are
    # never touched, so the iPhone never lights up.
    if detailed:
        builtin_idx = [i for i, d in enumerate(detailed)
                       if "builtin" in (d.get("device_type") or "").lower()]
        for i in builtin_idx:
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                cap.release()
                continue
            # Single quick frame to confirm the device actually works. We
            # deliberately do NOT call _fingerprint_camera here — its 320×240
            # SET + warm-up adds delay and is unnecessary now that PyObjC has
            # told us this is the built-in.
            ok, frame = False, None
            for _ in range(40):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    break
                time.sleep(0.05)
            if ok and frame is not None and frame.size:
                name = detailed[i].get("name") or "name unknown"
                log.info("WINNER (PyObjC BuiltIn): cv2[%s] (%s) — iPhone indices not touched.",
                         i, name)
                return _configure(cap), i
            log.warning("PyObjC says cv2[%s] is BuiltIn but it never delivered a frame. "
                        "Falling through.", i)
            cap.release()

    # FAST PATH: a pinned cv2 index from settings.json. Open it, fingerprint
    # once, and short-circuit if it scores clearly built-in. This keeps the
    # iPhone from waking up on every subsequent start — we never touch it.
    if preferred is not None:
        idx = int(preferred)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            name = camera_name(idx) or "name unknown"
            device_type = (detailed[idx]["device_type"]
                           if idx < len(detailed) else "")
            info = _fingerprint_camera(cap, label=f"cv2[{idx}] {name} (pinned)")
            score = _score_as_builtin(info, device_type=device_type)
            log.info("Pinned cv2[%s] score=%d device_type=%s",
                     idx, score, device_type or "?")
            if info["frames_ok"] and score >= 50:
                t_ff = info.get("first_frame_ms", -1)
                log.info("WINNER (fast path): cv2[%s] · score=%d · first_frame=%dms — "
                         "skipping full probe so iPhone isn't woken up.",
                         idx, score, int(t_ff))
                return _configure(cap), idx
            log.info("Pinned cv2[%s] didn't pass the fast path (score=%d). "
                     "Falling through to full probe.", idx, score)
            cap.release()
        else:
            cap.release()

    # 2. Probe every working cv2 index and score it. SHORT-CIRCUIT on a clear
    # built-in (score ≥ 100) so we don't poke the iPhone if a better camera
    # has already been found.
    candidates = []  # list of (score, idx, info, cap)
    for i in range(C.MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        name = camera_name(i) or "name unknown"
        device_type = (detailed[i]["device_type"]
                       if i < len(detailed) else "")
        info = _fingerprint_camera(cap, label=f"cv2[{i}] {name}")
        score = _score_as_builtin(info, device_type=device_type)
        log.info("cv2 index %s (%s) score=%d device_type=%s",
                 i, name, score, device_type or "?")
        if info["frames_ok"] == 0:
            cap.release()
            continue
        if score >= 100:
            # Clearly the built-in. Don't probe further cv2 indices.
            for sc, ci, ii, cc in candidates:
                cc.release()
            t_ff = info.get("first_frame_ms", -1)
            log.info("WINNER (short-circuit): cv2[%s] · score=%d · first_frame=%dms — "
                     "not probing remaining indices.", i, score, int(t_ff))
            return _configure(cap), i
        candidates.append((score, i, info, cap))

    if not candidates:
        log.error("No camera could be opened at all.")
        return None, None

    # Pick the highest-scoring camera. If there's a clear iPhone (score < 0),
    # we'll never end up on it unless it's the only thing that works.
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_idx, best_info, best_cap = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None

    # Honour `preferred` ONLY if its score ties the best score (i.e. it's a
    # safe choice). Otherwise the user's stale pin doesn't override us.
    if preferred is not None:
        for score, idx, info, cap in candidates:
            if idx == int(preferred) and score >= best_score - 5:
                # Close enough — release the actually-best one, use the pin.
                if idx != best_idx:
                    best_cap.release()
                    best_score, best_idx, best_info, best_cap = score, idx, info, cap
                break

    # Release every camera we opened except the winner.
    for score, idx, info, cap in candidates:
        if idx != best_idx:
            cap.release()

    name = camera_name(best_idx) or "name unknown"
    t_ff = best_info.get("first_frame_ms", -1)
    likely_label = "built-in" if t_ff < 300 else ("iPhone-like" if t_ff > 800 else "ambiguous")
    log.info("WINNER: cv2[%s] · score=%d · first_frame=%dms (%s) · default=%dx%d · low=%dx%d",
             best_idx, best_score, int(t_ff), likely_label,
             best_info["default_w"], best_info["default_h"],
             best_info["low_res_w"], best_info["low_res_h"])
    log.info("AVFoundation label for cv2[%s] is %r — but ENUMERATION ORDER may differ; "
             "behavioral fingerprint is what we trust.", best_idx, name)
    if best_score < 0:
        log.warning("Best camera scored negative — all candidates look iPhone-like. "
                    "If wrong, set GAZE_CAMERA_INDEX=N or hit the Pick button.")
    if runner_up:
        rs, ri, ri_info, _ = runner_up
        log.info("Runner-up was cv2[%s] · score=%d · first_frame=%dms.",
                 ri, rs, int(ri_info.get("first_frame_ms", -1)))

    return _configure(best_cap), best_idx


def open_camera_index(i):
    """Open a specific camera index, or return None."""
    cap = cv2.VideoCapture(int(i))
    if cap.isOpened():
        return _configure(cap)
    cap.release()
    return None


@dataclass
class FrameResult:
    """Outcome of analysing a single frame."""
    face_found: bool
    focused: bool
    state: FocusState
    bad_ratio: float                 # 1 - attention (5-min EWMA), for the gauge / history
    dominant: FocusState
    attention: float = 0.0           # smoothed 0..1 attention level (5-min EWMA)
    warn_bad_ratio: float = 0.0      # 1 - attention from the short (60s) warning EWMA
    annotated: object = None         # BGR numpy frame with overlay, for live preview


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GazeAnalyzer:
    """Owns the FaceMesh model + rolling focus window + calibrated baseline."""

    def __init__(self):
        self._fm = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.base_eye = None
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._focus_history = deque()      # (ts, bool)  — legacy binary path
        self._state_history = deque()      # (ts, FocusState) — for dominant-state
        self._inner_score = 0.0            # 5-min EWMA on per-frame soft score (gauge / history)
        self._warn_inner_score = 0.0       # 60-s EWMA on per-frame soft score (warning trigger)
        self._attention = 0.0              # outer-EMA-smoothed display level
        self._eye_closed_since = None      # timestamp eyes first dropped below thresh
        self._last_s_eye_open = 1.0        # last soft-eye-open score; held during blinks
        self._last_s_gaze = 1.0            # last soft-gaze score; iris unreliable when closed

    # ---- Calibration ----
    def calibrate(self, cap, say=lambda *a, **k: None, should_stop=lambda: False,
                  on_frame=lambda *a, **k: None):
        """Guided calibration. `say(key, text, priority=False)` delivers spoken
        hints. `on_frame(annotated_bgr)` receives every captured frame so the
        sidebar preview stays live during calibration (otherwise the user
        can't position themselves). Returns True on success. The timer only
        advances while the face is well positioned."""
        say("calib_intro",
            "Let's get you set up. Settle into your normal work position, "
            "just as you would naturally.", priority=True)
        time.sleep(3.0)
        say("calib_pos",
            "Make sure your face is in the camera's view — you don't need to look at it.")

        eyes, yaws, pitches = [], [], []
        last_hint_text = ""
        last_hint_time = 0.0
        HINT_COOLDOWN = 2.5
        last_progress = -1
        good_start = None
        bad_streak = 0
        deadline = time.time() + C.CALIB_DEADLINE

        def _draw_calib_overlay(frame, face_found, progress=None, hint=None):
            """Tiny orange border + progress bar at the bottom + optional hint."""
            out = frame.copy()
            h, w = out.shape[:2]
            color = (60, 200, 240) if face_found else (60, 120, 240)
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 3)
            label = "CALIBRATING"
            cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if progress is not None:
                bar_w = int(max(0.0, min(1.0, progress / 100.0)) * (w - 20))
                cv2.rectangle(out, (10, h - 20), (10 + bar_w, h - 10), color, -1)
            if hint:
                cv2.putText(out, hint[:38], (10, h - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            return out

        while time.time() < deadline and not should_stop():
            ok, f = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            r = self._fm.process(rgb)
            now = time.time()
            face_found = bool(r.multi_face_landmarks)

            if not face_found:
                bad_streak += 1
                if bad_streak >= C.CALIB_BAD_STREAK_RESET:
                    good_start = None
                    eyes.clear(); yaws.clear(); pitches.clear()
                    last_progress = -1
                if now - last_hint_time > HINT_COOLDOWN:
                    say("no_face_calib",
                        "I can't see your face yet. Try adjusting the camera "
                        "angle until you're in frame.", priority=True)
                    last_hint_text = "no_face"
                    last_hint_time = now
                on_frame(_draw_calib_overlay(f, False, hint="No face yet"))
                continue

            lm = r.multi_face_landmarks[0].landmark
            EAR_L, EAR_R, L, R = get_eye_features(lm, *f.shape[1::-1])
            hints = get_calib_hints(lm, *f.shape[1::-1], EAR_L, EAR_R)

            if hints:
                bad_streak += 1
                hint = hints[0]
                if bad_streak >= C.CALIB_BAD_STREAK_RESET:
                    good_start = None
                    eyes.clear(); yaws.clear(); pitches.clear()
                    last_progress = -1
                if hint != last_hint_text and now - last_hint_time > HINT_COOLDOWN:
                    say(f"hint_{hint}", hint, priority=True)
                    last_hint_text = hint
                    last_hint_time = now
                on_frame(_draw_calib_overlay(f, True, hint=hint))
                continue

            bad_streak = 0
            if good_start is None:
                good_start = now
                say("hold_still", "Perfect. Stay just like that.")
                last_hint_text = "hold_still"
                last_hint_time = now

            elapsed_good = now - good_start
            eye = (L + R) / 2
            yaw, pitch = get_head_offset(lm, *f.shape[1::-1])
            eyes.append(eye); yaws.append(yaw); pitches.append(pitch)

            progress = int((elapsed_good / C.CALIB_SECONDS) * 100)
            for milestone, line in {25: "Almost there, keep going.",
                                    50: "Halfway through.",
                                    75: "Nearly done."}.items():
                if last_progress < milestone <= progress:
                    say(f"prog_{milestone}", line)
            last_progress = progress
            on_frame(_draw_calib_overlay(f, True, progress=progress))

            if elapsed_good >= C.CALIB_SECONDS:
                break

        if len(eyes) < C.CALIB_MIN_SAMPLES:
            say("calib_fail",
                "That didn't quite work. Make sure your face is in view and "
                "we'll try again.", priority=True)
            return False

        self.base_eye = np.median(eyes, 0)
        self.base_yaw = float(np.median(yaws))
        self.base_pitch = float(np.median(pitches))
        self._smooth_yaw = self.base_yaw
        self._smooth_pitch = self.base_pitch
        self._focus_history.clear()
        self._state_history.clear()
        # Seed the inner EWMA + outer EMA at 1.0 — after calibration we trust
        # the user is focused, so the gauge starts at 100%. From here, the
        # EWMA will decay naturally if the user actually drifts.
        self._inner_score = 1.0
        self._warn_inner_score = 1.0
        self._attention = 1.0
        self._eye_closed_since = None
        self._last_s_eye_open = 1.0
        self._last_s_gaze = 1.0
        say("calib_done",
            "All set. I've learned your natural working position and I'll keep "
            "an eye on things from here.", priority=True)
        return True

    # ---- Per-frame classification ----
    def _classify(self, lm, w, h, yaw, pitch):
        """Binary state classification (kept for the dominant-state machine and
        the on-screen overlay) + continuous soft score.

        Blink handling: spontaneous blinks (eyes closed for ≲400 ms) are NOT
        drowsiness. While EAR is below threshold but the closure hasn't lasted
        DROWSY_MIN_CLOSE_SEC yet, we (a) keep the state as FOCUSED instead of
        DROWSY, (b) skip the gaze-displacement check (iris is hidden so its
        position is meaningless), and (c) hold the eye-open and gaze sigmoid
        scores at their last computed value so the soft_score (which feeds the
        inner EWMA) doesn't dip on every blink.
        """
        EAR_L, EAR_R, L, R = get_eye_features(lm, w, h)
        eye = (L + R) / 2
        ear_min = min(EAR_L, EAR_R)
        d_yaw = abs(yaw - self.base_yaw)
        d_pitch = abs(pitch - self.base_pitch)
        eye_disp = float(np.linalg.norm(eye - self.base_eye))

        eyes_open_hard = EAR_L > C.EAR_OPEN_THRESH and EAR_R > C.EAR_OPEN_THRESH
        head_ok = d_yaw < C.HEAD_YAW_THRESH and d_pitch < C.HEAD_PITCH_THRESH
        eye_ok = eye_disp < C.EYE_REL_THRESH

        # Track how long the eyes have been closed.
        now = time.time()
        if eyes_open_hard:
            self._eye_closed_since = None
            eye_closed_for = 0.0
        else:
            if self._eye_closed_since is None:
                self._eye_closed_since = now
            eye_closed_for = now - self._eye_closed_since
        sustained_drowsy = eye_closed_for >= C.DROWSY_MIN_CLOSE_SEC
        in_blink = (not eyes_open_hard) and (not sustained_drowsy)

        # Head soft scores are always reliable.
        s_yaw = _sigmoid(((C.HEAD_YAW_THRESH + C.YAW_SOFT_RELAX) - d_yaw) / C.YAW_SOFT_K)
        s_pitch = _sigmoid(((C.HEAD_PITCH_THRESH + C.PITCH_SOFT_RELAX) - d_pitch) / C.PITCH_SOFT_K)

        # Eye-open + gaze scores: hold at last good value during a blink. When
        # eyes are open (or sustained-closed → drowsy) we recompute and cache.
        if in_blink:
            s_eye_open = self._last_s_eye_open
            s_gaze = self._last_s_gaze
        else:
            s_eye_open = _sigmoid((ear_min - (C.EAR_OPEN_THRESH - C.EAR_SOFT_RELAX)) / C.EAR_SOFT_K)
            s_gaze = _sigmoid(((C.EYE_REL_THRESH + C.EYE_SOFT_RELAX) - eye_disp) / C.EYE_SOFT_K)
            self._last_s_eye_open = s_eye_open
            self._last_s_gaze = s_gaze

        soft_score = float(s_eye_open * s_yaw * s_pitch * s_gaze)

        # State cascade. Order matters: drowsy beats head-turn beats looking-away.
        if sustained_drowsy:
            return False, FocusState.DROWSY, soft_score
        if not head_ok:
            return False, FocusState.HEAD_TURNED, soft_score
        if in_blink:
            # Brief blink — treat as focused (gaze direction unknown).
            return True, FocusState.FOCUSED, soft_score
        if not eye_ok:
            return False, FocusState.LOOKING_AWAY, soft_score
        return True, FocusState.FOCUSED, soft_score

    def analyze(self, frame, draw=True):
        """Analyse one BGR frame and update the rolling window."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        now = time.time()
        cutoff = now - C.FOCUS_WINDOW_SECONDS

        if not res.multi_face_landmarks:
            # Don't hard-reset the EWMA. Treat each no-face frame as fully
            # distracted (soft_score = 0) and let the inner/outer smoothers
            # decay naturally — so when the user steps back in, the gauge
            # already reflects how long they were gone instead of jumping
            # from 0 back to 100.
            self._focus_history.append((now, False))
            self._state_history.append((now, FocusState.NO_FACE))
            while self._focus_history and self._focus_history[0][0] < cutoff:
                self._focus_history.popleft()
            while self._state_history and self._state_history[0][0] < cutoff:
                self._state_history.popleft()

            if C.CONTINUOUS_SCORING:
                a_inner = C.INNER_EWMA_ALPHA
                self._inner_score = (1.0 - a_inner) * self._inner_score  # + α·0
                a_warn = C.WARN_INNER_EWMA_ALPHA
                self._warn_inner_score = (1.0 - a_warn) * self._warn_inner_score
                target = max(1e-6, C.ATTENTION_FULL_RATIO)
                raw_attention = min(1.0, self._inner_score / target)
                a_outer = C.ATTENTION_EMA_ALPHA
                self._attention = (a_outer * raw_attention +
                                   (1.0 - a_outer) * self._attention)
            else:
                # Binary fallback path: use the windowed good-ratio.
                self._attention = (sum(1 for _, x in self._focus_history if x) /
                                   max(1, len(self._focus_history)))

            self._eye_closed_since = None
            # _last_s_eye_open / _last_s_gaze stay put so a returning face
            # doesn't open with a 0-valued first frame.

            bad_states = [s for _, s in self._state_history if s != FocusState.FOCUSED]
            dominant = (Counter(bad_states).most_common(1)[0][0]
                        if bad_states else FocusState.NO_FACE)
            bad_ratio = 1.0 - self._attention
            warn_attention = min(
                1.0, self._warn_inner_score / max(1e-6, C.ATTENTION_FULL_RATIO))
            warn_bad_ratio = 1.0 - warn_attention

            annotated = self._draw(frame, FocusState.NO_FACE, bad_ratio) if draw else None
            return FrameResult(False, False, FocusState.NO_FACE, bad_ratio,
                               dominant, self._attention, warn_bad_ratio, annotated)

        lm = res.multi_face_landmarks[0].landmark
        raw_yaw, raw_pitch = get_head_offset(lm, w, h)
        self._smooth_yaw = (C.YAW_SMOOTH_ALPHA * raw_yaw +
                            (1 - C.YAW_SMOOTH_ALPHA) * self._smooth_yaw)
        self._smooth_pitch = (C.PITCH_SMOOTH_ALPHA * raw_pitch +
                              (1 - C.PITCH_SMOOTH_ALPHA) * self._smooth_pitch)

        focused, state, soft_score = self._classify(
            lm, w, h, self._smooth_yaw, self._smooth_pitch)

        self._focus_history.append((now, focused))
        self._state_history.append((now, state))
        while self._focus_history and self._focus_history[0][0] < cutoff:
            self._focus_history.popleft()
        while self._state_history and self._state_history[0][0] < cutoff:
            self._state_history.popleft()

        if C.CONTINUOUS_SCORING:
            # Inner EWMA on the soft per-frame score — gives recency weighting,
            # so a long working session is what gets reflected in the gauge.
            a_inner = C.INNER_EWMA_ALPHA
            self._inner_score = (a_inner * soft_score +
                                 (1.0 - a_inner) * self._inner_score)
            # Parallel short-window EWMA the warning state machine watches —
            # 60 s half-life so warnings fire after ~2 min of acute distraction.
            a_warn = C.WARN_INNER_EWMA_ALPHA
            self._warn_inner_score = (a_warn * soft_score +
                                      (1.0 - a_warn) * self._warn_inner_score)
            target = max(1e-6, C.ATTENTION_FULL_RATIO)
            raw_attention = min(1.0, self._inner_score / target)
            # Outer EMA is purely cosmetic — keeps the UI gauge from twitching.
            a_outer = C.ATTENTION_EMA_ALPHA
            self._attention = (a_outer * raw_attention +
                               (1.0 - a_outer) * self._attention)
            attention = self._attention
            warn_attention = min(1.0, self._warn_inner_score / target)
        else:
            good_ratio = (sum(1 for _, x in self._focus_history if x) /
                          max(1, len(self._focus_history)))
            attention = good_ratio
            self._attention = attention
            warn_attention = good_ratio

        bad_ratio = 1.0 - attention
        warn_bad_ratio = 1.0 - warn_attention
        bad_states = [s for _, s in self._state_history if s != FocusState.FOCUSED]
        dominant = (Counter(bad_states).most_common(1)[0][0]
                    if bad_states else FocusState.FOCUSED)

        annotated = self._draw(frame, state, bad_ratio) if draw else None
        return FrameResult(True, focused, state, bad_ratio, dominant,
                           attention, warn_bad_ratio, annotated)

    # ---- Live preview overlay ----
    @staticmethod
    def _draw(frame, state, bad_ratio):
        colors = {
            FocusState.FOCUSED: (80, 220, 100),
            FocusState.DROWSY: (60, 160, 240),
            FocusState.LOOKING_AWAY: (60, 200, 240),
            FocusState.HEAD_TURNED: (60, 120, 240),
            FocusState.NO_FACE: (130, 130, 130),
        }
        out = frame.copy()
        h, w = out.shape[:2]
        col = colors.get(state, (200, 200, 200))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 4)
        label = state.value.replace("_", " ").upper()
        cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        # focus bar
        bar_w = int((1.0 - bad_ratio) * (w - 20))
        cv2.rectangle(out, (10, h - 20), (10 + bar_w, h - 10), col, -1)
        return out

    def close(self):
        try:
            self._fm.close()
        except Exception:
            pass


class WarningController:
    """Hysteresis state machine driving warnings, decoupled from any output.

    `notifier` must provide:
        speak(key, text, cooldown=..., priority=...)
        warn_audio_start()
        warn_audio_stop()
        on_enter_warning(state)         # for logging a distraction event
        on_exit_warning(state)
    """

    # Three tiers per state, in order of escalation:
    #   tier 1 — gentle, single nudge at ~2 min total distraction
    #   tier 2 — firmer, around ~5 min total
    #   tier 3 — more present, ~10 min+ total (a real break is in order)
    _MESSAGES = {
        FocusState.DROWSY: (
            "Your eyes look a little heavy — take a breath whenever you're ready.",
            "You've been resting your eyes for a few minutes. A short break could help.",
            "It's been over ten minutes of heavy eyes. Please take a real break — "
            "stand up, drink some water, come back when you feel refreshed.",
        ),
        FocusState.LOOKING_AWAY: (
            "Whenever you're ready, your work is here for you.",
            "You've been away from your work for a few minutes. Come back when you can.",
            "It's been over ten minutes now. Take a proper break if you need one, "
            "then come back focused.",
        ),
        FocusState.HEAD_TURNED: (
            "Something has your attention — that's okay, take your time.",
            "You've been turned away for a few minutes. Ready to come back?",
            "Over ten minutes turned away. Whatever it is can wait, or take a real "
            "break before coming back.",
        ),
        FocusState.NO_FACE: (
            "I can't quite see you — that's okay, take your time.",
            "Still can't see you. Come back when you're ready.",
            "It's been over ten minutes — checking in. I'm here whenever you return.",
        ),
    }
    _DEFAULT_MESSAGES = (
        "Looks like attention drifted a bit — come back when you're ready.",
        "You've been unfocused for a few minutes. Try to bring your attention back.",
        "It's been over ten minutes of drifting. Take a proper break if you need one.",
    )

    def __init__(self, notifier):
        self._n = notifier
        self._in_warning = False
        self._warn_start = None
        self._warn_state = None
        # Track the last (state, escalation) combo we actually emitted so
        # that repeated update() calls every frame don't keep adding the
        # same utterance to the TTS queue.
        self._last_emit_key = None

    @property
    def in_warning(self):
        return self._in_warning

    def update(self, bad_ratio, dominant_state, no_face):
        # `no_face` argument kept for backward compat but ignored — analyzer
        # now feeds normal bad_ratio + NO_FACE dominant state during no-face.
        del no_face
        if not self._in_warning:
            if bad_ratio >= C.WARN_ENTER_RATIO:
                self._in_warning = True
                self._warn_start = time.time()
                self._warn_state = dominant_state
                self._last_emit_key = None  # force speak on entry
                self._n.on_enter_warning(dominant_state)
                self._emit(dominant_state)
        else:
            if bad_ratio < C.WARN_EXIT_RATIO:
                self._exit(speak_recovery=True)
            else:
                self._emit(dominant_state)

    def _emit(self, state):
        # Pick a tier from how long we've been in warning. Tier 1 covers
        # ~2-5 min of total distraction (gentle), tier 2 covers ~5-10 min
        # (firmer), tier 3 takes over at 10 min+ (more present).
        elapsed = time.time() - self._warn_start
        if elapsed >= C.WARN_TIER3_SEC:
            tier = 3
            chime_gap = C.ALERT_MIN_INTERVAL_TIER3
            cooldown = 45.0
        elif elapsed >= C.WARN_TIER2_SEC:
            tier = 2
            chime_gap = C.ALERT_MIN_INTERVAL_TIER2
            cooldown = 60.0
        else:
            tier = 1
            chime_gap = C.ALERT_MIN_INTERVAL
            cooldown = 90.0
        self._n.warn_audio_start(min_interval=chime_gap)
        key = f"warn_{state.value}_t{tier}"
        # Only enqueue when the (state, tier) pair actually changes. Without
        # this, every frame in warning re-runs speak() and the TTS queue would
        # accumulate utterances that keep playing after the user has refocused.
        if key == self._last_emit_key:
            return
        self._last_emit_key = key
        msgs = self._MESSAGES.get(state, self._DEFAULT_MESSAGES)
        text = msgs[tier - 1]
        self._n.speak(key, text, cooldown=cooldown)

    def _exit(self, speak_recovery):
        prev = self._warn_state
        self._in_warning = False
        self._warn_start = None
        self._warn_state = None
        self._last_emit_key = None
        self._n.warn_audio_stop()
        self._n.on_exit_warning(prev)
        if speak_recovery:
            # priority=True clears anything still queued + kills the currently-
            # playing utterance, so the user never hears a stale "you've been
            # drifting" line after they've already come back.
            self._n.speak("refocused",
                          "Welcome back. You're doing great.",
                          cooldown=0, priority=True)
