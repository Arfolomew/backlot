#!/bin/bash
# Put a Backlot icon in your app menu and on your desktop.
#
#   setup/install-launcher.sh            # install
#   setup/install-launcher.sh --remove   # take it back out
#
# Any environment start.sh understands can be baked into the launcher, e.g.
#   PORT=8802 COMFYUI_URL=http://127.0.0.1:8189 setup/install-launcher.sh
#
# Linux: writes a freedesktop .desktop entry plus the icon set, and a copy on
# ~/Desktop if that folder exists. macOS: writes Backlot.command on the Desktop
# (double-click to launch); the Finder icon is left as the default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="backlot"

# Env vars to carry into the launcher, only the ones actually set.
ENVSTR=""
for v in PORT PROXY_PORT COMFYUI_URL BACKLOT_QWEN_URL; do
  if [ -n "${!v:-}" ]; then ENVSTR="$ENVSTR $v=${!v}"; fi
done

case "$(uname -s)" in
  Linux)
    APPS="$HOME/.local/share/applications"
    ICONS="$HOME/.local/share/icons/hicolor"
    DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    if [ "${1:-}" = "--remove" ]; then
      rm -f "$APPS/$NAME.desktop" "$DESKTOP_DIR/$NAME.desktop"
      for s in 16 32 48 64 128 256 512; do rm -f "$ICONS/${s}x${s}/apps/$NAME.png"; done
      rm -f "$ICONS/scalable/apps/$NAME.svg"
      command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
      echo "Backlot launcher removed."
      exit 0
    fi
    for s in 16 32 48 64 128 256 512; do
      mkdir -p "$ICONS/${s}x${s}/apps"
      cp "$ROOT/app/icon/png/$NAME-$s.png" "$ICONS/${s}x${s}/apps/$NAME.png"
    done
    mkdir -p "$ICONS/scalable/apps" "$APPS"
    cp "$ROOT/app/icon/$NAME.svg" "$ICONS/scalable/apps/$NAME.svg"
    if [ -n "$ENVSTR" ]; then EXEC="env$ENVSTR $ROOT/setup/launch.sh"; else EXEC="$ROOT/setup/launch.sh"; fi
    cat > "$APPS/$NAME.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Backlot
Comment=Local video and image studio on your own ComfyUI
Exec=$EXEC
Icon=$NAME
Terminal=false
Categories=Graphics;
StartupNotify=true
DESKTOP
    chmod +x "$APPS/$NAME.desktop" "$ROOT/setup/launch.sh" "$ROOT/start.sh"
    command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
    command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$ICONS" 2>/dev/null || true
    echo "App menu: Backlot ($APPS/$NAME.desktop)"
    if [ -d "$DESKTOP_DIR" ]; then
      cp "$APPS/$NAME.desktop" "$DESKTOP_DIR/$NAME.desktop"
      chmod +x "$DESKTOP_DIR/$NAME.desktop"
      # GNOME marks new desktop launchers untrusted until you allow them once.
      command -v gio >/dev/null && gio set "$DESKTOP_DIR/$NAME.desktop" metadata::trusted true 2>/dev/null || true
      echo "Desktop: $DESKTOP_DIR/$NAME.desktop (if it shows as untrusted, right-click it and choose Allow Launching)"
    fi
    ;;
  Darwin)
    OUT="$HOME/Desktop/Backlot.command"
    if [ "${1:-}" = "--remove" ]; then rm -f "$OUT"; echo "Backlot launcher removed."; exit 0; fi
    printf '#!/bin/bash\n%s exec "%s/setup/launch.sh"\n' "${ENVSTR:+env$ENVSTR}" "$ROOT" > "$OUT"
    chmod +x "$OUT" "$ROOT/setup/launch.sh" "$ROOT/start.sh"
    echo "Desktop: $OUT (double-click to launch)"
    ;;
  *)
    echo "Unsupported OS: $(uname -s). Run ./start.sh directly." >&2; exit 1 ;;
esac
