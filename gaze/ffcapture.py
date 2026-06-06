"""ffmpeg-subprocess camera capture, used to bypass cv2 when its AVFoundation
backend doesn't enumerate the BuiltIn camera (a known issue with some
opencv-python wheels — anaconda's in particular — on macOS 14+).

Exposes a class that mimics enough of cv2.VideoCapture's interface
(`read`, `isOpened`, `release`, `set`, `get`) that the rest of the codebase
can use it as a drop-in replacement.

Selects the device by NAME (e.g. "FaceTime HD Camera") via ffmpeg's
avfoundation indev, so we don't have to know which AVFoundation index it
sits at — we just say "give me FaceTime HD" and ffmpeg handles it.
"""

import logging
import shutil
import subprocess
import threading
import time

import numpy as np

log = logging.getLogger("gaze")


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


class FFmpegCamera:
    """A minimal cv2.VideoCapture-compatible capture backed by an `ffmpeg
    -f avfoundation` subprocess. Selects the device by name.

    Usage:
        cap = FFmpegCamera("FaceTime HD Camera", width=320, height=240)
        if cap.isOpened():
            ok, frame = cap.read()  # frame is HxWx3 uint8 BGR
            ...
            cap.release()
    """

    # cv2 property IDs we honor (without depending on cv2 at import time).
    _CV2_PROP_W = 3       # cv2.CAP_PROP_FRAME_WIDTH
    _CV2_PROP_H = 4       # cv2.CAP_PROP_FRAME_HEIGHT
    _CV2_PROP_FPS = 5     # cv2.CAP_PROP_FPS

    def __init__(self, device_name, width=320, height=240, fps=30):
        self._name = device_name
        self._w = int(width)
        self._h = int(height)
        self._fps = int(fps)
        self._frame_bytes = self._w * self._h * 3
        self._lock = threading.Lock()
        self._latest = None
        self._stop = False
        self._proc = None
        self._reader_thread = None
        self._opened = False
        self._start()

    def _start(self):
        if not ffmpeg_available():
            log.warning("FFmpegCamera requested but ffmpeg is not on PATH.")
            return
        # Capture at a size the device actually supports (640×480 is the
        # smallest mode FaceTime HD on M-series Macs offers), then scale down
        # to the requested size with a video filter so the rest of the
        # pipeline can keep its existing 320×240 contract. fps must match a
        # mode the device exposes (FaceTime HD on M2 supports 30 fps natively;
        # 15 is matched-against-float and rejected as 15.000000 != 15).
        # Do NOT pin -pixel_format on the input side — pairing it with
        # specific (width, height, fps) often produces "Input/output error"
        # at open time even though the format is listed as supported. Let
        # ffmpeg pick a native format (nv12 / uyvy422) and convert internally.
        capture_w, capture_h = max(self._w, 640), max(self._h, 480)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", str(self._fps),
            "-video_size", f"{capture_w}x{capture_h}",
            "-i", self._name,
            "-vf", f"scale={self._w}:{self._h}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * self._frame_bytes,
            )
        except Exception:
            log.exception("Could not launch ffmpeg subprocess for FFmpegCamera")
            return

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        # Mark opened only if we got an early frame; the caller's first read()
        # will block until then anyway via isOpened()-then-read pattern.
        self._opened = True
        # Drain a bit of stderr so it doesn't block.
        threading.Thread(target=self._stderr_drain, daemon=True).start()

    def _reader_loop(self):
        if self._proc is None or self._proc.stdout is None:
            return
        bs = self._frame_bytes
        buf = bytearray()
        try:
            while not self._stop:
                chunk = self._proc.stdout.read(bs - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= bs:
                    raw = bytes(buf[:bs])
                    del buf[:bs]
                    try:
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                            self._h, self._w, 3)
                    except Exception:
                        continue
                    with self._lock:
                        self._latest = frame
        except Exception:
            log.exception("FFmpegCamera reader loop crashed")

    def _stderr_drain(self):
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            for line in iter(self._proc.stderr.readline, b""):
                if not line:
                    break
                s = line.decode("utf-8", errors="replace").strip()
                if not s:
                    continue
                # Log everything so we can diagnose camera-permission and
                # device-format issues; the user can pipe it elsewhere later.
                low = s.lower()
                if "error" in low or "fail" in low or "denied" in low:
                    log.warning("ffmpeg: %s", s)
                else:
                    log.info("ffmpeg: %s", s)
        except Exception:
            pass

    # ---------- cv2-compatible API ----------
    def isOpened(self):
        if not self._opened or self._proc is None:
            return False
        return self._proc.poll() is None

    def read(self, *_args, **_kw):
        # Block briefly until we have a first frame, then return latest.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return True, self._latest.copy()
            if not self.isOpened():
                return False, None
            time.sleep(0.02)
        with self._lock:
            if self._latest is not None:
                return True, self._latest.copy()
        return False, None

    def release(self):
        self._stop = True
        self._opened = False
        try:
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def set(self, prop, _val):
        # ffmpeg can't be reconfigured mid-stream; size was baked at start.
        return False

    def get(self, prop):
        if prop == self._CV2_PROP_W:
            return float(self._w)
        if prop == self._CV2_PROP_H:
            return float(self._h)
        if prop == self._CV2_PROP_FPS:
            return float(self._fps)
        return 0.0

    def getBackendName(self):
        return "FFMPEG_AVFOUNDATION"
