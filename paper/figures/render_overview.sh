#!/usr/bin/env bash
# Render overview_figure.html -> overview_figure.png (2x, trimmed) and overview_figure.pdf
# Requires: Google Chrome (headless) and ImageMagick (`magick`).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML="$DIR/overview_figure.html"
PNG="$DIR/overview_figure.png"
PDF="$DIR/overview_figure.pdf"
TMP="$DIR/.overview_raw.png"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Figure canvas size (must match .fig width/height in the HTML).
W=1200
H=1500

"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
  --default-background-color=FFFFFFFF \
  --force-device-scale-factor=2 \
  --window-size="${W},${H}" \
  --screenshot="$TMP" \
  "file://$HTML" >/dev/null 2>&1

# Crop surrounding whitespace, then add a small uniform white margin.
magick "$TMP" -trim +repage -bordercolor white -border 24 "$PNG"
rm -f "$TMP"

"$CHROME" --headless=new --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$PDF" \
  "file://$HTML" >/dev/null 2>&1

echo "Wrote:"
echo "  $PNG"
echo "  $PDF"
