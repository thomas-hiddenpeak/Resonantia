#!/usr/bin/env bash
# Experimental voice training (pure C++/CUDA runtime, no Python):
#   raw recordings -> sliced clips -> fine-tuned decoder -> retrieval index.
# Only WAV input is currently supported. Keep the source recordings: training
# quality/stability is not yet a production guarantee.
#
# Usage:
#   scripts/train_voice.sh --raw <wav|dir> --work <dir> [options]
#
# Options:
#   --raw <path>    Raw target-voice recording(s): a WAV or a WAV-only directory
#   --work <dir>    Working directory for clips + outputs (created if missing)
#   --models <dir>  Model root (default: <repo>/models)
#   --build <dir>   Build directory with the CLIs (default: <repo>/build)
#   --steps <n>     Fine-tune steps (default: 300)
#   --seg-sec <f>   Clip length in seconds (default: 3.0)
#   --lr <f>        Learning rate (default: 2e-4)
#   --speaker <id>  Source speaker embedding id (default: 0)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$REPO/models"
BUILD="$REPO/build"
RAW=""; WORK=""; STEPS=300; SEG_SEC=3.0; LR=2e-4; SPEAKER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw) RAW="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --build) BUILD="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --seg-sec) SEG_SEC="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --speaker) SPEAKER="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | tail -n +2 | sed 's/^# \?//'; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

[[ -n "$RAW" && -n "$WORK" ]] || { echo "error: --raw and --work are required" >&2; exit 1; }
[[ -e "$RAW" ]] || { echo "error: input does not exist: $RAW" >&2; exit 1; }

if [[ -f "$RAW" && "${RAW,,}" != *.wav ]]; then
  echo "error: only WAV input is supported: $RAW" >&2
  exit 1
fi
if [[ -d "$RAW" ]]; then
  UNSUPPORTED="$(find "$RAW" -type f \( -iname '*.flac' -o -iname '*.mp3' \) -print -quit)"
  if [[ -n "$UNSUPPORTED" ]]; then
    echo "error: directory contains unsupported audio (WAV only): $UNSUPPORTED" >&2
    exit 1
  fi
  FIRST_WAV="$(find "$RAW" -type f -iname '*.wav' -print -quit)"
  if [[ -z "$FIRST_WAV" ]]; then
    echo "error: directory contains no WAV files: $RAW" >&2
    exit 1
  fi
fi

HUBERT="$MODELS/hubert_base/model.safetensors"
RMVPE="$MODELS/rmvpe.safetensors"
PRETRAINED="$MODELS/pretrained_v2/pretrained_v2/f0G40k.safetensors"
for f in "$HUBERT" "$RMVPE" "$PRETRAINED"; do
  [[ -f "$f" ]] || { echo "error: missing model: $f" >&2; exit 1; }
done
for exe in vc_preprocess vc_train build_index; do
  [[ -x "$BUILD/$exe" ]] || { echo "error: missing executable: $BUILD/$exe" >&2; exit 1; }
done

CLIPS="$WORK/clips"
MODEL_OUT="$WORK/model.safetensors"
INDEX_OUT="$WORK/model.index"
mkdir -p "$WORK"

echo "==> [1/3] Slicing $RAW -> $CLIPS"
"$BUILD/vc_preprocess" --input "$RAW" --output-dir "$CLIPS" \
  --sr 40000 --seg-sec "$SEG_SEC" --trim

echo "==> [2/3] Experimental decoder fine-tune ($STEPS steps) -> $MODEL_OUT"
"$BUILD/vc_train" --hubert "$HUBERT" --rmvpe "$RMVPE" --pretrained "$PRETRAINED" \
  --target "$CLIPS" --out "$MODEL_OUT" --steps "$STEPS" --lr "$LR" --speaker "$SPEAKER"

echo "==> [3/3] Building retrieval index -> $INDEX_OUT"
"$BUILD/build_index" --hubert "$HUBERT" --input-dir "$CLIPS" --output "$INDEX_OUT"

echo
echo "Done. Convert with:"
echo "  scripts/convert_voice.sh --model $MODEL_OUT --index $INDEX_OUT \\"
echo "    --input <song.wav> --output <out.wav>"
