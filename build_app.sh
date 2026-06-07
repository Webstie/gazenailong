#!/usr/bin/env bash
# build_app.sh — rebuild dist/Gaze Nailong.app and dist/Gaze Nailong.dmg.
#
# Usage:   ./build_app.sh
# Output:  dist/Gaze Nailong.app   (drag-to-Applications)
#          dist/Gaze Nailong.dmg   (shareable installer)
#
# The .app is signed ad-hoc (no Apple Developer ID). First-launch Gatekeeper
# will say "Apple could not verify Gaze Nailong is free of malicious software".
# Right-click → Open once, or run:
#     xattr -dr com.apple.quarantine "/Applications/Gaze Nailong.app"

set -euo pipefail

APP_NAME="Gaze Nailong"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"

if ! "$PY" -m PyInstaller --version >/dev/null 2>&1; then
    echo "PyInstaller not found in $PY — installing..."
    "$PY" -m pip install --upgrade pyinstaller
fi

echo "→ Cleaning build/ and dist/"
rm -rf build dist

echo "→ Running PyInstaller"
"$PY" -m PyInstaller --noconfirm GazeNailong.spec

APP="dist/${APP_NAME}.app"
if [ ! -d "$APP" ]; then
    echo "PyInstaller did not produce $APP" >&2
    exit 1
fi

# Some macOS file providers (iCloud Drive, Documents folder sync, etc.) keep
# restamping com.apple.FinderInfo onto anything inside the project tree, and
# codesign refuses bundles with that xattr. Work around it by relocating the
# bundle to /tmp, signing there, then moving back. `ditto --norsrc --noextattr`
# guarantees a clean copy with no xattrs / resource forks.
STAGE="$(mktemp -d -t gaze-build.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
CLEAN_APP="$STAGE/${APP_NAME}.app"

echo "→ Relocating bundle to $STAGE to strip xattrs"
ditto --norsrc --noextattr "$APP" "$CLEAN_APP"

echo "→ Ad-hoc signing"
codesign --force --deep --sign - "$CLEAN_APP"
codesign --verify --verbose "$CLEAN_APP"

echo "→ Moving signed bundle back to dist/"
rm -rf "$APP"
ditto --norsrc --noextattr "$CLEAN_APP" "$APP"
# Re-sign in place — the move can re-introduce FinderInfo on the destination
# parent, so we sign once more after the file provider has had its way with
# the bundle. (The signature itself is path-independent, but Gatekeeper looks
# happier with a freshly-stamped one.)
codesign --force --deep --sign - "$APP" 2>/dev/null || true

echo "→ Building dist/${APP_NAME}.dmg"
DMG_STAGE="$STAGE/dmg"
mkdir -p "$DMG_STAGE"
ditto --norsrc --noextattr "$CLEAN_APP" "$DMG_STAGE/${APP_NAME}.app"
ln -s /Applications "$DMG_STAGE/Applications"
DMG_TMP="$STAGE/${APP_NAME}.dmg"
hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGE" \
    -ov -format UDZO -fs HFS+ \
    "$DMG_TMP" >/dev/null
rm -f "dist/${APP_NAME}.dmg"
mv "$DMG_TMP" "dist/${APP_NAME}.dmg"

echo
echo "✓ Built:"
echo "    $APP"
echo "    dist/${APP_NAME}.dmg ($(du -h "dist/${APP_NAME}.dmg" | cut -f1))"
