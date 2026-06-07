"""Background monitor: ties detector + audio + storage together.

Runs the capture/analyze loop on its own thread. Exposes a thread-safe
status snapshot (incl. an on-screen guidance `message`) and the latest
annotated JPEG frame. Handles calibration, pause/resume, recalibration,
camera switching, and a voice on/off toggle.

Notifications: speech is OFF by default (it sounded unnatural and the engine
interrupted itself). Distractions are signalled by a single rate-limited
chime plus the on-screen `message`; voice can be re-enabled via set_voice().
"""

import os
import json
import time
import logging
import threading

import cv2

log = logging.getLogger("gaze")

from . import config as C
from .detector import (GazeAnalyzer, WarningController, FocusState,
                       open_camera, open_camera_index,
                       blocked_camera_indices, refresh_camera_inventory,
                       camera_name as _camera_name_lookup)
from .audio import AudioManager
from .storage import Store, DB_DIR

SETTINGS_PATH = os.path.join(DB_DIR, "settings.json")


class _Notifier:
    """Bridges WarningController -> on-screen message + chime + event logging."""

    def __init__(self, monitor, store):
        self._m = monitor
        self._store = store

    def speak(self, key, text, cooldown=3.0, priority=False):
        # routes through the monitor so it sets the on-screen message and only
        # actually speaks if voice is enabled
        self._m._say(key, text, cooldown=cooldown, priority=priority)

    def warn_audio_start(self, min_interval=None):
        self._m._audio.warn_audio_start(min_interval=min_interval)

    def warn_audio_stop(self):
        self._m._audio.warn_audio_stop()

    def on_enter_warning(self, state):
        self._store.open_event(state.value if hasattr(state, "value") else state)

    def on_exit_warning(self, state):
        self._store.close_event()


class Monitor:
    """Public API: start, pause, resume, toggle, recalibrate, switch_camera,
    set_voice, stop, status, latest_jpeg."""

    def __init__(self, alert_path=None):
        self._audio = AudioManager(alert_path=alert_path)
        self._store = Store()
        self._notifier = _Notifier(self, self._store)

        self._voice = C.VOICE_DEFAULT
        self._cam_index = None
        self._load_settings()

        self._analyzer = None
        self._warn = None
        self._cap = None

        self._thread = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._recalib = threading.Event()
        self._switch = threading.Event()
        self._target_index = None    # if set, _handle_switch opens THIS index

        self._lock = threading.Lock()
        self._jpeg = None
        self._assessment = None   # see start_assessment()
        self._status = {
            "running": False,
            "calibrating": False,
            "paused": False,
            "state": "idle",
            "focused": False,
            "face_found": False,
            "bad_ratio": 0.0,
            "attention": 0.0,
            "in_warning": False,
            "no_camera": False,
            "message": "",
            "camera_index": self._cam_index,
            "camera_name": None,
            "camera_resolution": None,
            "voice": self._voice,
            "tts_backend": self._audio.tts_backend,
            "assessment": {"active": False, "completed": False},
        }
        self._last_sample = 0.0

    # ---------- settings ----------
    def _load_settings(self):
        if os.environ.get("GAZE_VOICE", "").lower() in ("1", "true", "yes"):
            self._voice = True
        try:
            with open(SETTINGS_PATH) as f:
                d = json.load(f)
            self._voice = bool(d.get("voice", self._voice))
            ci = d.get("camera_index")
            self._cam_index = int(ci) if ci is not None else None
        except Exception:
            pass
        # If a previous run pinned a Continuity Camera index, drop it so
        # open_camera() can re-probe and pick the built-in this time.
        if self._cam_index is not None and self._cam_index in blocked_camera_indices():
            log.info("Ignoring saved camera_index=%s (Continuity Camera).",
                     self._cam_index)
            self._cam_index = None
        # The -1 sentinel means "ffmpeg-managed" — there's no real cv2 index
        # to honor. Treat as no preference so open_camera retries ffmpeg fresh.
        if self._cam_index is not None and self._cam_index < 0:
            self._cam_index = None

    def _save_settings(self):
        try:
            os.makedirs(DB_DIR, exist_ok=True)
            with open(SETTINGS_PATH, "w") as f:
                json.dump({"voice": self._voice,
                           "camera_index": self._cam_index}, f)
        except Exception:
            log.exception("Could not save settings")

    # ---------- status / preview ----------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def status(self):
        with self._lock:
            s = dict(self._status)
        s["today"] = self._store.today_summary()
        a = self._assessment
        if a is not None:
            remaining = max(0.0, a["end_ts"] - time.time()) if a.get("active") else 0.0
            s["assessment"] = {
                "active": bool(a.get("active")),
                "completed": bool(a.get("completed")),
                "ended_early": bool(a.get("ended_early")),
                "duration_sec": a.get("duration_sec"),
                "remaining_sec": round(remaining, 1),
                "session_id": a.get("session_id"),
            }
        return s

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def _publish_frame(self, annotated):
        if annotated is None:
            return
        ok, buf = cv2.imencode(".jpg", annotated,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def _say(self, key, text, cooldown=3.0, priority=False):
        """Set the on-screen guidance message; speak only if voice is on."""
        self._set(message=text)
        if self._voice:
            self._audio.speak(key, text, cooldown=cooldown, priority=priority)

    # ---------- lifecycle ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        self._paused.set()
        self._audio.warn_audio_stop()
        self._set(paused=True, message="Paused")

    def resume(self):
        self._paused.clear()
        self._set(paused=False, message="")

    def recalibrate(self):
        self._recalib.set()

    def switch_camera(self):
        self._target_index = None
        self._switch.set()

    def set_camera_index(self, idx):
        """Switch to a specific cv2 index on the next loop iteration. Used by
        the sidebar camera picker so the user can pick visually."""
        try:
            self._target_index = int(idx)
        except (TypeError, ValueError):
            self._target_index = None
            return
        self._switch.set()

    def set_voice(self, on):
        self._voice = bool(on)
        if not self._voice:
            self._audio.warn_audio_stop()
        self._set(voice=self._voice)
        self._save_settings()
        return self._voice

    def toggle(self):
        if self._paused.is_set():
            self.resume()
        else:
            self.pause()

    # ---------- timed assessment ----------
    def start_assessment(self, duration_sec):
        """Begin a fixed-duration focus session.

        Behaviour during a session:
          * the warning state machine is suppressed (no chimes, no voice
            interruptions), but the gauge and history still update
          * a fresh store session is opened with kind='assessment' so the
            report only reflects this window
        On completion the user gets a single chime + voice line that bypasses
        the voice toggle (the user explicitly asked to be told when the
        session ends).
        """
        try:
            duration_sec = int(duration_sec)
        except (TypeError, ValueError):
            return None
        duration_sec = max(60, min(7200, duration_sec))  # 1 min – 2 h

        # Bail out of any active warning so it can't echo into the session.
        if self._warn is not None and self._warn.in_warning:
            try:
                self._warn._exit(speak_recovery=False)
            except Exception:
                pass
        self._audio.warn_audio_stop()

        # Roll over the store session: end whatever's open, then start a
        # fresh one tagged as an assessment. The report endpoint queries
        # samples/events by this session_id.
        self._store.end_session()
        sid = self._store.start_session(kind="assessment",
                                        target_seconds=duration_sec)
        now = time.time()
        self._assessment = {
            "active": True,
            "completed": False,
            "ended_early": False,
            "start_ts": now,
            "end_ts": now + duration_sec,
            "duration_sec": duration_sec,
            "session_id": sid,
        }
        minutes = int(round(duration_sec / 60))
        self._set(message=f"{minutes}-minute focus session started — I'll be quiet.",
                  in_warning=False)
        log.info("Assessment started: %d s (session %s)", duration_sec, sid)
        return self._assessment

    def stop_assessment(self, ended_early=True):
        """Wrap up the current assessment. `ended_early=False` is the natural
        timer-expiry path (announces completion); `True` is the user-cancel
        path."""
        if not self._assessment or not self._assessment.get("active"):
            return None
        a = self._assessment
        a["active"] = False
        a["completed"] = True
        a["ended_early"] = bool(ended_early)
        a["completed_ts"] = time.time()

        # Close the assessment store session. Resume a normal session so the
        # monitor keeps tracking after the assessment finishes.
        self._store.end_session(ended_early=ended_early)
        self._store.start_session(kind="normal")

        minutes = int(round(a["duration_sec"] / 60))
        if ended_early:
            text = "Session ended early. Take care."
        else:
            text = f"Your {minutes}-minute focus session is complete. Nice work."

        self._set(message=text, in_warning=False)
        # Completion announcement bypasses the voice toggle (we want the user
        # to know the session is over even if they had voice muted).
        try:
            self._audio.warn_audio_start(min_interval=0.0)
        except Exception:
            pass
        try:
            self._audio.speak("assessment_done", text, cooldown=0, priority=True)
        except Exception:
            pass
        log.info("Assessment %s (session %s)",
                 "ended early" if ended_early else "completed",
                 a.get("session_id"))
        return a

    def dismiss_assessment(self):
        """Clear the completed-assessment banner once the user has acked it
        (e.g. opened the report). The session_id is preserved on the side so
        the report can still be fetched, but `completed` flips back to False
        so the sidebar UI returns to its normal layout."""
        if self._assessment and self._assessment.get("completed"):
            self._assessment["completed"] = False
        self._set(message="")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._cleanup()

    def _cleanup(self):
        try:
            self._store.end_session()
        except Exception:
            pass
        self._audio.stop()
        if self._analyzer:
            self._analyzer.close()
        if self._cap:
            self._cap.release()
        self._set(running=False, state="stopped")

    # ---------- camera switching ----------
    def _open_next_camera(self):
        """Open the next working cv2 index (wrapping), excluding only the
        current one. We deliberately do *not* apply the Continuity-Camera
        blocklist here — Switch is a manual override, so the user can land
        on any camera, including the iPhone, if our auto-detection ever
        picked the wrong one."""
        refresh_camera_inventory()
        cur = self._cam_index if self._cam_index is not None else -1
        candidates = []
        for off in range(1, C.MAX_CAMERA_INDEX + 1):
            idx = (cur + off) % C.MAX_CAMERA_INDEX
            if idx == cur:
                continue
            candidates.append(idx)
        for idx in candidates:
            cap = open_camera_index(idx)
            if cap is not None:
                return cap, idx
        return None, None

    # ---------- main loop ----------
    def _run(self):
        try:
            self._run_inner()
        except Exception as e:  # never let the monitor thread die silently
            log.exception("Monitor crashed")
            self._set(running=False, state="error", error=str(e))
            try:
                self._cleanup()
            except Exception:
                log.exception("Cleanup after crash failed")

    def _publish_camera_info(self):
        """Make the active camera's name + native resolution visible in
        /api/status. ffmpeg-managed captures tag themselves with `_gaze_name`."""
        name = getattr(self._cap, "_gaze_name", None) if self._cap is not None else None
        if name is None and self._cam_index is not None and self._cam_index >= 0:
            name = _camera_name_lookup(self._cam_index)
        res = None
        try:
            if self._cap is not None:
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if w and h:
                    res = f"{w}×{h}"
        except Exception:
            pass
        self._set(camera_name=name, camera_resolution=res)

    def _run_inner(self):
        self._cap, self._cam_index = open_camera(self._cam_index)
        if self._cap is None:
            self._set(running=False, no_camera=True, state="no_camera",
                      message="No camera found — connect one and restart.")
            return

        self._analyzer = GazeAnalyzer()
        self._warn = WarningController(self._notifier)
        self._set(running=True, no_camera=False, camera_index=self._cam_index)
        self._publish_camera_info()
        name = (getattr(self._cap, "_gaze_name", None)
                or _camera_name_lookup(self._cam_index)
                or "unknown")
        log.info("Using camera index %s (%s)", self._cam_index, name)
        # Save the resolved index so next run starts on the same camera.
        self._save_settings()

        if not self._do_calibration():
            self._cleanup()
            return

        self._store.start_session()

        while not self._stop.is_set():
            # Timed assessment auto-completion: fires the one-time chime +
            # voice line and rolls over to a normal session.
            if self._assessment and self._assessment.get("active"):
                if time.time() >= self._assessment["end_ts"]:
                    self.stop_assessment(ended_early=False)

            if self._switch.is_set():
                self._switch.clear()
                # A camera switch resets calibration → end any active
                # assessment first (its baseline would no longer apply).
                if self._assessment and self._assessment.get("active"):
                    self.stop_assessment(ended_early=True)
                self._handle_switch()
                if self._stop.is_set():
                    break

            if self._recalib.is_set():
                self._recalib.clear()
                if self._assessment and self._assessment.get("active"):
                    self.stop_assessment(ended_early=True)
                self._store.end_session()
                if not self._do_calibration():
                    break
                self._store.start_session()

            if self._paused.is_set():
                self._set(state="paused")
                time.sleep(0.15)
                continue

            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            try:
                result = self._analyzer.analyze(frame, draw=True)
            except Exception:
                log.exception("Frame analysis error (skipping frame)")
                time.sleep(0.05)
                continue
            self._publish_frame(result.annotated)

            now = time.time()
            # No special-case for face_found here any more — analyzer keeps the
            # EWMA decaying during no-face, and WarningController fires the
            # "I can't see you" message naturally once attention drops past
            # WARN_ENTER_RATIO. That removes both the ugly 0% cliff and the
            # redundant 2-second NO_FACE prompt timer.
            assessment_active = bool(self._assessment and
                                     self._assessment.get("active"))
            if not assessment_active:
                self._warn.update(result.warn_bad_ratio, result.dominant,
                                  no_face=False)
                in_warning_now = self._warn.in_warning
            else:
                # During a timed assessment: do not run the warning state
                # machine at all. The user explicitly asked for no
                # interference, so we suppress chimes / voice / banner.
                in_warning_now = False
            self._set(state=result.state.value,
                      focused=result.focused,
                      face_found=result.face_found,
                      bad_ratio=round(result.bad_ratio, 3),
                      attention=round(result.attention, 3),
                      in_warning=in_warning_now,
                      paused=False)
            if assessment_active or (not in_warning_now and result.face_found):
                self._set(message="")
            if now - self._last_sample >= C.SAMPLE_LOG_INTERVAL:
                self._last_sample = now
                self._store.log_sample(result.attention,
                                        result.dominant.value)

            time.sleep(0.05)

        self._cleanup()

    def _handle_switch(self):
        target = self._target_index
        self._target_index = None  # one-shot
        if target is not None and target == self._cam_index:
            return  # already on it; nothing to do
        if target is not None:
            # Release current first so the target index can claim the device.
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            newcap = open_camera_index(target)
            newidx = target if newcap is not None else None
            if newcap is None:
                # Reopen the previous one so the monitor isn't left blind.
                newcap, newidx = open_camera(self._cam_index)
                if newcap is None:
                    self._say("cam_none",
                              f"Camera {target} could not be opened.",
                              priority=True)
                    return
                self._say("cam_none",
                          f"Camera {target} could not be opened.",
                          priority=True)
        else:
            newcap, newidx = self._open_next_camera()
            if newcap is None:
                self._say("cam_none", "No other camera found.", priority=True)
                return
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = newcap
        self._cam_index = newidx
        self._set(camera_index=newidx)
        self._publish_camera_info()
        self._save_settings()
        log.info("Switched to camera index %s (%s)",
                 newidx, _camera_name_lookup(newidx))
        self._say("cam_switched",
                  f"Switched to camera {newidx}. Recalibrating.", priority=True)
        self._store.end_session()
        if self._do_calibration():
            self._store.start_session()

    def _do_calibration(self):
        self._set(calibrating=True, state="calibrating")
        ok = False
        while not ok and not self._stop.is_set():
            ok = self._analyzer.calibrate(
                self._cap,
                say=self._say,
                should_stop=lambda: self._stop.is_set() or self._switch.is_set(),
                on_frame=self._publish_frame,
            )
            if not ok and not self._stop.is_set():
                if self._switch.is_set():
                    break
                self._say("retry_calib",
                          "Let's try the setup again in a moment.", priority=True)
                time.sleep(3.0)
        self._set(calibrating=False)
        return ok and not self._stop.is_set()
