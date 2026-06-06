"""Local Flask dashboard server.

Serves the dashboard UI, a JSON status/history API, and an MJPEG live
preview of the annotated camera frame. Everything stays on 127.0.0.1 —
the video never leaves the machine.
"""

import base64
import os
import threading
import time
import webbrowser

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from . import config as C
from .detector import _macos_camera_inventory

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASH_URL = f"http://{C.SERVER_HOST}:{C.SERVER_PORT}/"


def create_app(monitor):
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(_HERE, "dashboard.html")

    @app.route("/sidebar")
    def sidebar():
        return send_from_directory(_HERE, "sidebar.html")

    @app.route("/api/open-browser", methods=["POST"])
    def open_browser():
        try:
            webbrowser.open(_DASH_URL)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/status")
    def status():
        return jsonify(monitor.status())

    @app.route("/api/history")
    def history():
        store = monitor._store
        return jsonify({
            "timeline": store.recent_samples(since_seconds=900),
            "breakdown": store.distraction_breakdown(since_seconds=86400),
            "daily": store.daily_focus(days=7),
            "today": store.today_summary(),
        })

    @app.route("/api/control/<action>", methods=["POST"])
    def control(action):
        if action == "pause":
            monitor.pause()
        elif action == "resume":
            monitor.resume()
        elif action == "recalibrate":
            monitor.recalibrate()
        elif action == "switch-camera":
            monitor.switch_camera()
        else:
            return jsonify({"ok": False, "error": "unknown action"}), 400
        return jsonify({"ok": True})

    @app.route("/api/voice/<state>", methods=["POST"])
    def voice(state):
        on = state.lower() in ("on", "1", "true", "yes")
        return jsonify({"ok": True, "voice": monitor.set_voice(on)})

    @app.route("/api/cameras/probe")
    def cameras_probe():
        """Opens each cv2 index 0..MAX-1 and grabs one frame; returns base64
        JPEG thumbnails so the sidebar can show a visual picker. The currently
        active index reuses the monitor's last annotated frame (we can't open
        the same device twice on macOS)."""
        inv = _macos_camera_inventory()
        cur = monitor.status().get("camera_index")
        out = []
        for i in range(C.MAX_CAMERA_INDEX):
            name = inv[i] if i < len(inv) else None
            jpeg_b64 = None
            try:
                if i == cur:
                    j = monitor.latest_jpeg()
                    if j:
                        jpeg_b64 = base64.b64encode(j).decode()
                else:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        # warm-up: macOS often returns a black first frame
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
                pass
            if jpeg_b64 is not None:
                out.append({
                    "index": i,
                    "name": name,
                    "current": (i == cur),
                    "image": jpeg_b64,
                })
        return jsonify({"cameras": out, "current": cur})

    @app.route("/api/shutdown", methods=["POST"])
    def shutdown():
        """Stops the monitor and exits the process so Chrome + Flask both quit."""
        def _bye():
            try:
                monitor.stop()
            except Exception:
                pass
            os._exit(0)
        # Give Flask a moment to flush the response, then hard-exit.
        threading.Timer(0.25, _bye).start()
        return jsonify({"ok": True})

    @app.route("/api/cameras/select", methods=["POST"])
    def cameras_select():
        body = request.get_json(silent=True) or {}
        try:
            idx = int(body.get("index"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid index"}), 400
        monitor.set_camera_index(idx)
        return jsonify({"ok": True, "index": idx})

    @app.route("/video")
    def video():
        def gen():
            boundary = b"--frame"
            while True:
                jpeg = monitor.latest_jpeg()
                if jpeg is not None:
                    yield (boundary + b"\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.06)
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def run_server(monitor, host=C.SERVER_HOST, port=C.SERVER_PORT):
    app = create_app(monitor)
    # threaded so the MJPEG stream doesn't block API calls
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
