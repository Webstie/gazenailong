"""All tunable constants for Gaze Nailong, in one place.

These values are lifted verbatim from the original main.py so the desktop
build behaves identically to the Orange Pi build.
"""

# -------- FaceMesh landmark indices --------
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362

IRIS_POINTS = list(range(468, 478))

FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152
NOSE_TIP = 1

# -------- Calibration --------
CALIB_SECONDS = 5.0          # seconds of *good* frames required
CALIB_MIN_SAMPLES = 30
CALIB_FACE_SIZE_MIN = 0.20   # face_width / frame_width
CALIB_BAD_STREAK_RESET = 10  # consecutive bad frames before resetting progress
CALIB_DEADLINE = 30.0        # hard give-up after this many seconds

# -------- Detection thresholds --------
EAR_OPEN_THRESH = 0.20
EYE_REL_THRESH = 0.12
HEAD_YAW_THRESH = 0.15
HEAD_PITCH_THRESH = 0.15

# Eyes briefly closed (typical spontaneous blink: 100–400 ms) shouldn't be
# treated as drowsiness. Only count eye closure as DROWSY after this many
# seconds of sustained closure. During the under-threshold window we also
# hold the eye-open and gaze sigmoid scores at their last good value, since
# the iris position isn't reliably detectable through closed eyelids.
DROWSY_MIN_CLOSE_SEC = 0.6

# -------- Pose smoothing (EMA dead band) --------
YAW_SMOOTH_ALPHA = 0.15
PITCH_SMOOTH_ALPHA = 0.15

# -------- Focus scoring --------
# When True, each frame produces a continuous score in [0,1] from soft (sigmoid)
# thresholds, the score feeds an EWMA (the "inner" smoother), and the result is
# rescaled so that ATTENTION_FULL_RATIO equivalent of perfect-focus counts as
# 100% attention. The display value goes through a second, faster EMA so the
# UI gauge doesn't jitter. When False, the old binary per-frame logic runs
# (using FOCUS_WINDOW_SECONDS as a true window).
CONTINUOUS_SCORING = True

FOCUS_WINDOW_SECONDS = 60.0      # still used by the binary fallback path
ATTENTION_FULL_RATIO = 0.667     # EWMA value that maps to 100% attention

# Inner EWMA on the per-frame soft score. α = ln(2) / (t_half * fps) so at
# 30 fps with t_half = 10 s → α ≈ 0.00231. That gives:
#   * half-life       ≈ 10 s   (drift 10 s ago → 50% weight now)
#   * time constant τ ≈ 14.4 s (drift τ s ago → 1/e ≈ 37% weight now)
#   * equivalent SMA  ≈ 29 s
# Smaller α = smoother but slower to reflect a change; larger α = snappier
# but jumpier. This is what makes "30s focused + 30s distracted" score
# differently from "30s distracted + 30s focused" — the older half of the
# window contributes exponentially less.
INNER_EWMA_ALPHA = 0.00231

ATTENTION_EMA_ALPHA = 0.15       # outer EMA, purely cosmetic (anti-jitter)

# Sigmoid soft-threshold parameters. RELAX shifts the inflection point inside
# the "focused" zone so being right at the hard threshold still scores ~0.7
# instead of 0.5 — that's the "relax" the user wanted. K controls how sharp the
# transition is around the inflection point.
EAR_SOFT_K = 0.03
EAR_SOFT_RELAX = 0.02
YAW_SOFT_K = 0.05
YAW_SOFT_RELAX = 0.05
PITCH_SOFT_K = 0.05
PITCH_SOFT_RELAX = 0.05
EYE_SOFT_K = 0.04
EYE_SOFT_RELAX = 0.04

# -------- Warning state machine --------
# Driven by `1 - attention` so the existing 0.60 / 0.30 hysteresis means
# "warn when attention < 40%, recover when attention > 70%".
WARN_ENTER_RATIO = 0.60
WARN_EXIT_RATIO = 0.30
WARN_ESCALATE_SEC = 15.0
NO_FACE_PROMPT_SEC = 2.0

# -------- Camera --------
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CAMERA_PROBE_ORDER = (0, 1, 2, 3)  # desktop webcams are usually index 0
MAX_CAMERA_INDEX = 6               # how many indices to scan when switching

# -------- Notifications --------
# Speech (TTS) sounded unnatural and the engine interrupted itself, so voice
# is OFF by default: distractions are signalled by a short sound + on-screen
# text. Flip with the sidebar "Voice" toggle or GAZE_VOICE=1.
VOICE_DEFAULT = False
ALERT_MIN_INTERVAL = 12.0    # min seconds between alert chimes while distracted

# -------- Behavior logging --------
SAMPLE_LOG_INTERVAL = 5.0    # seconds between focus-ratio samples written to DB

# -------- Dashboard server --------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8733
