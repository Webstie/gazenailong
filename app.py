#!/usr/bin/env python3
"""Gaze Nailong — desktop edition.

A focus monitor that runs quietly in the background. The default UI is a
slim **sidebar window** — opened as a chromeless Chrome/Edge "app window"
(reliable on every Python build) — with a live gauge, camera preview,
today's stats, and Pause / Recalibrate buttons. A button on the sidebar
opens the **full web dashboard** (charts + history) in your browser.

Run:
    python app.py             # sidebar app-window + dashboard server (default)
    python app.py --webview   # native pywebview window (needs framework Python)
    python app.py --tray      # menu-bar / tray icon instead of the sidebar
    python app.py --browser   # no window; just open the dashboard in a browser
    python app.py --headless  # monitor only, no UI (Orange-Pi-like)

The Orange Pi build now lives in orangepi/ and is untouched.
"""

import os
import sys
import time
import logging
import threading
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))


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
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # quiet Flask access logs
    logging.info("Log file: %s", log_path)
    return log_path


_setup_logging()
log = logging.getLogger("gaze")

from gaze import config as C
from gaze.monitor import Monitor
from gaze.server import run_server

DASH_URL = f"http://{C.SERVER_HOST}:{C.SERVER_PORT}/"
SIDEBAR_URL = f"http://{C.SERVER_HOST}:{C.SERVER_PORT}/sidebar"
WARNING_CANDIDATES = ["warning.wav", "warning.mp3", "warning.aiff"]


def _find_alert():
    for n in WARNING_CANDIDATES:
        p = os.path.join(BASE, n)
        if os.path.exists(p):
            return p
    return None


def _wait_for_server(timeout=8.0):
    """Block until the Flask server answers, so the window never loads a blank page."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(DASH_URL, timeout=0.5)
            return True
        except Exception:
            time.sleep(0.2)
    return False


# ---------- sidebar via a chromeless browser app-window (default) ----------
def _chromium_path():
    """Locate a Chromium-based browser that supports --app windows."""
    if sys.platform == "darwin":
        cands = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    elif sys.platform.startswith("win"):
        roots = [os.environ.get("PROGRAMFILES", ""),
                 os.environ.get("PROGRAMFILES(X86)", ""),
                 os.environ.get("LOCALAPPDATA", "")]
        cands = []
        for b in roots:
            if not b:
                continue
            cands += [
                os.path.join(b, r"Google\Chrome\Application\chrome.exe"),
                os.path.join(b, r"Microsoft\Edge\Application\msedge.exe"),
                os.path.join(b, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            ]
    else:
        from shutil import which
        for n in ("google-chrome", "chromium", "chromium-browser",
                  "microsoft-edge", "brave-browser"):
            p = which(n)
            if p:
                return p
        return None
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def run_app_window(monitor):
    """Open the sidebar as a chromeless app-window; fall back to a normal tab."""
    import subprocess
    chrome = _chromium_path()
    if chrome:
        # A dedicated profile dir forces a fresh Chrome instance, so the size
        # flags are honoured instead of being forwarded to a running Chrome
        # (which is why the window opened huge before). The page then docks
        # itself to the right edge via window.moveTo/resizeTo.
        profile = os.path.join(os.path.expanduser("~"), ".gazenailong", "chrome")
        try:
            subprocess.Popen(
                [chrome, f"--app={SIDEBAR_URL}",
                 f"--user-data-dir={profile}",
                 "--window-size=294,460",
                 "--no-first-run", "--no-default-browser-check"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("Sidebar opened as an app window (%s)", os.path.basename(chrome))
        except Exception:
            log.exception("Could not launch app window; opening a browser tab")
            webbrowser.open(SIDEBAR_URL)
    else:
        log.info("No Chromium browser found; opening the sidebar in your default browser")
        webbrowser.open(SIDEBAR_URL)
    log.info("Monitor running. Press Ctrl-C here to stop.")
    _idle_until_interrupt(monitor)


# ---------- native pywebview window (optional, --webview) ----------
def run_sidebar(monitor):
    import webview

    class _Api:
        def open_dashboard(self):
            webbrowser.open(DASH_URL)

    # Compact floating widget, top-right corner.
    W, H, MARGIN, TOP_GAP = 294, 460, 18, 56
    x = None
    try:
        scr = webview.screens[0]
        x = max(0, scr.width - W - MARGIN)
    except Exception:
        pass

    kw = dict(url=SIDEBAR_URL, js_api=_Api(), width=W, height=H,
              on_top=True, resizable=False, frameless=True)
    if x is not None:
        kw["x"], kw["y"] = x, TOP_GAP

    webview.create_window("Gaze Nailong", **kw)
    log.info("Opening sidebar window")
    t0 = time.time()
    webview.start()  # should block on the main thread until the window closes
    elapsed = time.time() - t0

    # On a non-framework Python build (e.g. Anaconda), the macOS Cocoa event
    # loop can't start and webview.start() returns instantly. Detect that and
    # report False so the caller can fall back to the browser dashboard.
    if elapsed < 2.0:
        log.warning(
            "Sidebar window exited after %.2fs — the GUI loop did not start. "
            "This is the classic non-framework-Python issue on macOS. "
            "Fix: `conda install -y python.app` then run `pythonw app.py`, "
            "or just use `python app.py --browser`.", elapsed)
        return False
    return True


# ---------- tray (optional, --tray) ----------
def run_tray(monitor):
    import pystray
    from pystray import MenuItem as Item
    from PIL import Image, ImageDraw

    def _icon(color=(86, 211, 100)):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((6, 18, 58, 46), outline=color, width=5)
        d.ellipse((26, 24, 40, 40), fill=color)
        return img

    def paused_state(_):
        return monitor.status().get("paused", False)

    menu = pystray.Menu(
        Item(lambda i: "Resume" if paused_state(i) else "Pause",
             lambda i, it: monitor.toggle()),
        Item("Recalibrate", lambda i, it: monitor.recalibrate()),
        Item("Open Dashboard", lambda i, it: webbrowser.open(DASH_URL), default=True),
        pystray.Menu.SEPARATOR,
        Item("Quit", lambda i, it: (monitor.stop(), i.stop())),
    )
    icon = pystray.Icon("gaze_nailong", _icon(), "Gaze Nailong", menu)

    def updater():
        last = None
        while True:
            s = monitor.status()
            col = (248, 81, 73) if s.get("in_warning") else \
                  (86, 211, 100) if s.get("focused") else \
                  (240, 136, 62) if s.get("running") else (130, 130, 130)
            if col != last:
                try:
                    icon.icon = _icon(col)
                except Exception:
                    pass
                last = col
            time.sleep(1.0)

    threading.Thread(target=updater, daemon=True).start()
    log.info("Starting tray icon")
    icon.run()


def _idle_until_interrupt(monitor):
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()


def _reset_history_db():
    """Each app start begins with a fresh history. Settings (camera index,
    voice toggle) survive in settings.json — only the focus log is wiped."""
    db = os.path.join(os.path.expanduser("~"), ".gazenailong", "history.db")
    try:
        os.remove(db)
        log.info("Cleared history DB at %s", db)
    except FileNotFoundError:
        pass
    except Exception:
        log.exception("Could not clear history DB at %s", db)


def main():
    args = set(sys.argv[1:])
    _reset_history_db()
    monitor = Monitor(alert_path=_find_alert())
    monitor.start()
    if getattr(monitor._audio, "voice_name", None):
        log.info("TTS voice: %s", monitor._audio.voice_name)

    if "--headless" in args:
        log.info("Running headless (Ctrl-C to stop).")
        _idle_until_interrupt(monitor)
        return

    # start the dashboard server in the background
    threading.Thread(target=run_server, args=(monitor,), daemon=True).start()
    _wait_for_server()
    log.info("Dashboard: %s", DASH_URL)

    if "--browser" in args:
        webbrowser.open(DASH_URL)
        _idle_until_interrupt(monitor)
        return

    if "--tray" in args:
        try:
            run_tray(monitor)
        except Exception:
            log.exception("Tray unavailable; falling back to browser")
            webbrowser.open(DASH_URL)
            _idle_until_interrupt(monitor)
        return

    if "--webview" in args:
        # native pywebview window — only reliable on a framework Python build
        ok = False
        try:
            ok = run_sidebar(monitor)
        except Exception:
            log.exception("pywebview window unavailable")
        if ok:
            monitor.stop()
        else:
            log.info("Falling back to an app window (Ctrl-C to stop).")
            run_app_window(monitor)
        return

    # default: chromeless browser app-window (reliable everywhere)
    run_app_window(monitor)


if __name__ == "__main__":
    main()
