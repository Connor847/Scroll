#!/bin/bash
# Downloads the MediaPipe Hands files into libs/mediapipe/ so the app runs offline.
# Run once:  bash download-mediapipe.sh

set -e
cd "$(dirname "$0")"            # work from this script's folder
DEST="libs/mediapipe"
mkdir -p "$DEST"

HANDS="https://cdn.jsdelivr.net/npm/@mediapipe/hands"
DRAW="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils"

files_hands=(
  hands.js
  hands.binarypb
  hands_solution_packed_assets.data
  hands_solution_packed_assets_loader.js
  hands_solution_simd_wasm_bin.js
  hands_solution_simd_wasm_bin.wasm
  hands_solution_simd_wasm_bin.data
  hands_solution_wasm_bin.js
  hands_solution_wasm_bin.wasm
  hand_landmark_full.tflite
  hand_landmark_lite.tflite
)

for f in "${files_hands[@]}"; do
  echo "Downloading $f"
  curl -fsSL "$HANDS/$f" -o "$DEST/$f"
done

echo "Downloading drawing_utils.js"
curl -fsSL "$DRAW/drawing_utils.js" -o "$DEST/drawing_utils.js"

echo ""
echo "Done. Files saved to $DEST :"
ls -lh "$DEST"
