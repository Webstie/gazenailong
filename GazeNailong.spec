# PyInstaller spec for Gaze Nailong.app
#
# Build:    pyinstaller --noconfirm GazeNailong.spec
# Output:   dist/Gaze Nailong.app
#
# After build, package into a DMG with build_app.sh.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

APP_NAME = "Gaze Nailong"
BUNDLE_ID = "com.gazenailong.app"
APP_VERSION = "1.0.2"

# ---------- data files bundled into the .app ----------
# Layout at runtime (sys._MEIPASS):
#   <MEI>/gaze/sidebar.html
#   <MEI>/gaze/report.html
#   <MEI>/gaze/dashboard.html
#   <MEI>/warning.wav
datas = [
    ("gaze/sidebar.html",   "gaze"),
    ("gaze/report.html",    "gaze"),
    ("gaze/dashboard.html", "gaze"),
    ("warning.wav",         "."),
]

# mediapipe ships .tflite/.binarypb models that its loader fetches by
# relative path; PyInstaller's hook does most of this but we explicitly
# collect to be safe across versions.
datas += collect_data_files("mediapipe")
datas += collect_data_files("cv2")

# ---------- hidden imports that static analysis misses ----------
hiddenimports = []
hiddenimports += collect_submodules("mediapipe")
hiddenimports += collect_submodules("pyttsx3")          # drivers loaded by name
hiddenimports += collect_submodules("webview")          # webview.platforms.cocoa
hiddenimports += collect_submodules("edge_tts")
hiddenimports += collect_submodules("aiohttp")
hiddenimports += [
    "pyttsx3.drivers.nsss",
    "objc",
    "Foundation",
    "AppKit",
    "WebKit",
    "AVFoundation",
    "Quartz",
    "CoreMedia",
]

# edge-tts ships SSL trust roots via certifi; aiohttp also loads them at
# import time. Bundle both so the cloud call works inside the sealed .app.
datas += collect_data_files("certifi")
datas += collect_data_files("edge_tts")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,             # --windowed: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=None,
    bundle_identifier=BUNDLE_ID,
    info_plist={
        "CFBundleName":              APP_NAME,
        "CFBundleDisplayName":       APP_NAME,
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion":           APP_VERSION,
        "LSMinimumSystemVersion":    "11.0",
        "NSHighResolutionCapable":   True,
        # Camera + mic prompts. Without these macOS silently denies access.
        "NSCameraUsageDescription":
            "Gaze Nailong uses the camera to detect whether you are looking at the screen.",
        "NSMicrophoneUsageDescription":
            "Gaze Nailong does not record audio; some macOS voice synthesis paths request mic access.",
        # We use AppleScript / NSSpeechSynthesizer for TTS, no Apple Events to
        # other apps, but declare just in case PyObjC tries.
        "NSAppleEventsUsageDescription":
            "Gaze Nailong does not control other apps.",
    },
)
