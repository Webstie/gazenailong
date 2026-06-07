#!/usr/bin/env python3
"""Gaze Nailong — native macOS app.

Runs the camera + analysis loop in the background and presents the UI inside
a native WKWebView window. Everything ships inside a single .app bundle —
no browser launches and no HTTP server.

UI ↔ Python communication is direct:
  * JS calls Python via pywebview's js_api bridge (see gaze/bridge.py).
  * Python pushes live frames into the window with evaluate_js() at ~4 fps.
"""

import base64
import logging
import os
import re
import sys
import threading
import time

# When frozen by PyInstaller, datas (HTML, warning.wav) are unpacked under
# sys._MEIPASS. In source mode they sit next to this file. Either way BASE
# is the directory that contains "gaze/" and "warning.wav".
if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
GAZE_DIR = os.path.join(BASE, "gaze")


# ---------- logging ----------
def _setup_logging():
    log_dir = os.path.join(os.path.expanduser("~"), ".gazenailong")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gaze.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log file: %s", log_path)
    return log_path


_setup_logging()
log = logging.getLogger("gaze")

import webview
from gaze.monitor import Monitor
from gaze.bridge import Bridge


WARNING_CANDIDATES = ["warning.wav", "warning.mp3", "warning.aiff"]
SIDEBAR_W, SIDEBAR_H = 294, 410


def _find_alert():
    for n in WARNING_CANDIDATES:
        p = os.path.join(BASE, n)
        if os.path.exists(p):
            return p
    return None


def _read_html(name):
    with open(os.path.join(GAZE_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


_REMOTE_ASSET = re.compile(
    r'<link[^>]*(?:fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*>'
    r'|<script[^>]+src="https?://[^"]+"[^>]*></script>',
    re.IGNORECASE,
)


def _strip_remote_assets(html):
    """Belt-and-braces: strip any remaining CDN <link>/<script> tags so the
    WebView is fully self-contained even if a future edit re-introduces a
    Google Fonts / jsdelivr reference. System fonts (-apple-system → SF Pro)
    render almost identically to the Inter we used to load remotely."""
    return _REMOTE_ASSET.sub("", html)


def _reset_history_db():
    """Per-launch fresh history (matches the previous Chrome-app behaviour)."""
    db = os.path.join(os.path.expanduser("~"), ".gazenailong", "history.db")
    try:
        os.remove(db)
        log.info("Cleared history DB at %s", db)
    except FileNotFoundError:
        pass
    except Exception:
        log.exception("Could not clear history DB at %s", db)


def _frame_pump(monitor):
    """Push the latest annotated JPEG into every open WebView window at
    ~4 fps. Replaces the old MJPEG /video endpoint."""
    while True:
        time.sleep(0.25)
        try:
            j = monitor.latest_jpeg()
            if not j or not webview.windows:
                continue
            b64 = base64.b64encode(j).decode()
            js = (f"window.setFrame && window.setFrame("
                  f"'data:image/jpeg;base64,{b64}')")
            for w in webview.windows:
                try:
                    w.evaluate_js(js)
                except Exception:
                    pass
        except Exception:
            log.exception("Frame pump iteration failed; continuing")


def _top_right_position():
    """Best-effort top-right docking. Returns (x, y) or (None, None) if the
    screen size is unknown at startup time."""
    try:
        scr = webview.screens[0]
        x = max(0, scr.width - SIDEBAR_W - 18)
        return x, 56
    except Exception:
        return None, None


def main():
    _reset_history_db()

    monitor = Monitor(alert_path=_find_alert())
    monitor.start()
    if getattr(monitor._audio, "voice_name", None):
        log.info("TTS voice: %s", monitor._audio.voice_name)

    sidebar_html = _strip_remote_assets(_read_html("sidebar.html"))
    report_html = _strip_remote_assets(_read_html("report.html"))

    bridge = Bridge(monitor)
    bridge._asset_html["report"] = report_html

    x, y = _top_right_position()
    kw = dict(
        html=sidebar_html,
        js_api=bridge,
        width=SIDEBAR_W, height=SIDEBAR_H,
        on_top=True, resizable=False, frameless=True,
    )
    if x is not None:
        kw["x"], kw["y"] = x, y

    sidebar = webview.create_window("Gaze Nailong", **kw)
    bridge._sidebar_window = sidebar

    threading.Thread(target=_frame_pump, args=(monitor,), daemon=True).start()

    def _on_closing():
        try:
            monitor.stop()
        except Exception:
            log.exception("Monitor.stop() during window close failed")
    try:
        sidebar.events.closing += _on_closing
    except Exception:
        # Older pywebview builds use a different event API — best-effort only.
        pass

    log.info("Starting WebView event loop")
    t0 = time.time()
    webview.start()  # blocks until all windows close
    elapsed = time.time() - t0

    # On a non-framework Python build (e.g. Anaconda), the macOS Cocoa loop
    # can't start and webview.start() returns instantly. Surface a clear
    # error so the dev fixes their environment instead of staring at silence.
    if elapsed < 2.0:
        log.error(
            "WebView event loop exited after %.2fs. This usually means you are "
            "running on a non-framework Python build (Anaconda is the usual "
            "culprit). Use the python.org installer or `conda install python.app` "
            "+ `pythonw app.py`. The packaged .app bundle ships its own Python "
            "and does not have this issue.", elapsed)

    log.info("All windows closed; exiting")
    try:
        monitor.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
