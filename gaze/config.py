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
# Iris/head tolerances are sized for a ~27" monitor at a typical 60 cm
# viewing distance: looking at the far corner of the screen (~25° off-axis)
# should still count as focused, not "looking away".
EAR_OPEN_THRESH = 0.20
EYE_REL_THRESH = 0.25
HEAD_YAW_THRESH = 0.20
HEAD_PITCH_THRESH = 0.20

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

# Inner EWMA on the per-frame soft score. α = ln(2) / (t_half * fps). At
# 30 fps with t_half = 300 s (5 min) → α ≈ 7.7e-5. That gives:
#   * half-life       ≈ 5 min   (drift 5 min ago → 50% weight now)
#   * time constant τ ≈ 7.2 min
#   * equivalent SMA  ≈ 14.4 min
# The gauge reflects sustained focus over a real work session rather than
# punishing every glance. Warnings use the separate, shorter EWMA below so
# they can still fire after a couple of minutes of acute distraction.
INNER_EWMA_ALPHA = 0.0000770

# Shorter EWMA used ONLY by the warning state machine. 60 s half-life at
# 30 fps → α = ln(2)/(60*30) ≈ 3.85e-4. With ATTENTION_FULL_RATIO = 0.667
# and WARN_ENTER_RATIO = 0.60, this fires after roughly 2 minutes of
# sustained distraction.
WARN_INNER_EWMA_ALPHA = 0.000385

ATTENTION_EMA_ALPHA = 0.15       # outer EMA, purely cosmetic (anti-jitter)

# Sigmoid soft-threshold parameters. RELAX shifts the inflection point inside
# the "focused" zone so being right at the hard threshold still scores ~0.7
# instead of 0.5 — that's the "relax" the user wanted. K controls how sharp the
# transition is around the inflection point.
EAR_SOFT_K = 0.03
EAR_SOFT_RELAX = 0.02
YAW_SOFT_K = 0.07
YAW_SOFT_RELAX = 0.07
PITCH_SOFT_K = 0.07
PITCH_SOFT_RELAX = 0.07
EYE_SOFT_K = 0.07
EYE_SOFT_RELAX = 0.07

# -------- Warning state machine --------
# Driven by `1 - warn_attention` (the short-window EWMA). With 60 s half-life
# and ATTENTION_FULL_RATIO = 0.667, the thresholds below mean:
#   * warn fires when warn_attention < 0.40  → ~2 min sustained distraction
#   * warn clears when warn_attention > 0.80 → ~45 s of clean refocus
WARN_ENTER_RATIO = 0.60
WARN_EXIT_RATIO = 0.20

# Three escalation tiers, measured from when the warning was entered:
#   tier 1  → on entry        — total ~2 min distraction, gentle nudge
#   tier 2  → +WARN_TIER2_SEC — total ~5 min, firmer
#   tier 3  → +WARN_TIER3_SEC — total ~10 min+, more present
WARN_TIER2_SEC = 180.0   # 3 min into warning ≈ 5 min total
WARN_TIER3_SEC = 480.0   # 8 min into warning ≈ 10 min total
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
# Min seconds between alert chimes, per escalation tier. Tier 1 is rare on
# purpose (one chime when the warning enters); tier 3 is more present so
# 10-minute+ distractions actually feel urgent.
ALERT_MIN_INTERVAL = 90.0          # tier 1 (≈ first 3 min of warning)
ALERT_MIN_INTERVAL_TIER2 = 45.0
ALERT_MIN_INTERVAL_TIER3 = 25.0

# -------- Behavior logging --------
SAMPLE_LOG_INTERVAL = 5.0    # seconds between focus-ratio samples written to DB

# -------- Dashboard server --------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8733
