"""JS-callable bridge exposed to the pywebview window.

Each method here is what the front-end calls as `pywebview.api.<name>(...)`.
The WebView and Python share the same process; pywebview marshals calls and
JSON-serialises return values automatically.
"""

import base64
import logging
import time

import cv2

log = logging.getLogger("gaze")


class Bridge:
    """Public façade over the Monitor for the WebView UI.

    All methods are safe to call from JavaScript. Heavy work is delegated
    to the Monitor (which runs on its own thread); these wrappers return
    quickly with JSON-friendly values.
    """

    def __init__(self, monitor):
        self._m = monitor
        # Underscore-prefixed so pywebview does NOT expose these as JS-callable
        # API methods. Filled in by app.py after window creation.
        self._sidebar_window = None
        self._asset_html = {}      # keys: 'report' (and 'dashboard' eventually)

    # ---------- status / live preview ----------
    def status(self):
        return self._m.status()

    def latest_frame(self):
        """Base64-encoded JPEG of the most recent annotated frame, or None.
        The sidebar polls this every ~250 ms so the preview stays live
        without the bandwidth of the old MJPEG endpoint."""
        j = self._m.latest_jpeg()
        if j is None:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(j).decode()

    # ---------- control ----------
    def pause(self):
        self._m.pause()
        return True

    def resume(self):
        self._m.resume()
        return True

    def toggle(self):
        self._m.toggle()
        return True

    def recalibrate(self):
        self._m.recalibrate()
        return True

    def switch_camera(self):
        self._m.switch_camera()
        return True

    def set_voice(self, on):
        return self._m.set_voice(bool(on))

    # ---------- voice picker ----------
    def voices_list(self):
        return self._m.list_voices()

    def voice_current(self):
        return self._m.voice_name()

    def voice_preview(self, name):
        try:
            return bool(self._m.preview_voice(str(name)))
        except Exception:
            return False

    def voice_set(self, name):
        try:
            return self._m.set_voice_name(str(name))
        except Exception:
            return None

    def set_camera_index(self, idx):
        try:
            self._m.set_camera_index(int(idx))
            return True
        except (TypeError, ValueError):
            return False

    # ---------- timed assessment ----------
    def assessment_start(self, duration_sec):
        try:
            dur = int(duration_sec)
        except (TypeError, ValueError):
            return None
        a = self._m.start_assessment(dur)
        if a is None:
            return None
        return {
            "session_id": a["session_id"],
            "duration_sec": a["duration_sec"],
            "end_ts": a["end_ts"],
        }

    def assessment_stop(self):
        a = self._m.stop_assessment(ended_early=True)
        if a is None:
            return None
        return {"session_id": a.get("session_id")}

    def assessment_dismiss(self):
        self._m.dismiss_assessment()
        return True

    def assessment_report(self, session_id):
        try:
            sid = int(session_id)
        except (TypeError, ValueError):
            return None
        return self._m._store.assessment_report(sid)

    # ---------- camera picker probe ----------
    def cameras_probe(self):
        """Open each cv2 index, grab one frame, return base64 thumbnails so
        the sidebar can show a visual picker. The active index reuses the
        monitor's most-recent annotated frame because macOS won't let us
        open the same device twice."""
        from .detector import _macos_camera_inventory
        from . import config as C

        inv = _macos_camera_inventory()
        cur = self._m.status().get("camera_index")
        out = []
        for i in range(C.MAX_CAMERA_INDEX):
            name = inv[i] if i < len(inv) else None
            jpeg_b64 = None
            try:
                if i == cur:
                    j = self._m.latest_jpeg()
                    if j:
                        jpeg_b64 = base64.b64encode(j).decode()
                else:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        for _ in range(3):
                            ok, frame = cap.read()
                            if ok and frame is not None:
                                break
                            time.sleep(0.05)
                        if ok and frame is not None:
                            frame = cv2.resize(frame, (160, 120))
                            ok, buf = cv2.imencode(
                                ".jpg", frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if ok:
                                jpeg_b64 = base64.b64encode(buf.tobytes()).decode()
                        cap.release()
            except Exception:
                log.exception("Camera probe failed for index %s", i)
            if jpeg_b64 is not None:
                out.append({
                    "index": i,
                    "name": name,
                    "current": (i == cur),
                    "image": jpeg_b64,
                })
        return {"cameras": out, "current": cur}

    # ---------- window spawning ----------
    def open_report(self, session_id):
        """Open the report in a fresh native window (no browser)."""
        try:
            sid = int(session_id)
        except (TypeError, ValueError):
            return False
        html = self._asset_html.get("report")
        if html is None:
            log.error("Report HTML not registered with bridge")
            return False
        # Pywebview allows window creation after start() — UI thread is handled
        # internally on macOS.
        import webview
        webview.create_window(
            "Focus Report — Gaze Nailong",
            html=html,
            js_api=self,
            width=900, height=920, resizable=True,
        )
        # Once the new window has finished loading we push the session id
        # into it. We poll briefly because pywebview doesn't expose a
        # `loaded` callback before .start() so we just retry via JS.
        import threading
        def _push():
            for _ in range(40):
                time.sleep(0.1)
                try:
                    last = webview.windows[-1]
                    last.evaluate_js(f"window.__sessionId = {sid}; "
                                     "if(window.loadReport) loadReport();")
                    return
                except Exception:
                    continue
        threading.Thread(target=_push, daemon=True).start()
        return True

    def quit_app(self):
        try:
            self._m.stop()
        except Exception:
            log.exception("Monitor.stop() failed during quit")
        import os
        # Bypass pywebview teardown — its event loop can hang on macOS if
        # the camera thread is mid-frame. Hard exit is the reliable path.
        os._exit(0)
