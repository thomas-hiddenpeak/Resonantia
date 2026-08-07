#!/usr/bin/env bash
# Convert audio with a fine-tuned voice model (+ optional retrieval index).
#
# Usage:
#   scripts/convert_voice.sh --model <G.safetensors> --input <wav> --output <wav> [options]
#
# Options:
#   --model <path>    Fine-tuned model from train_voice.sh
#   --input <wav>     Source audio to convert
#   --output <wav>    Output path
#   --index <path>    Retrieval index (optional)
#   --index-rate <f>  Index blend 0..1 (default: 0.5 when --index is given)
#   --pitch <semi>    Pitch shift in semitones (default: 0)
#   --models <dir>    Model root (default: <repo>/models)
#   --build <dir>     Build directory (default: <repo>/build)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$REPO/models"; BUILD="$REPO/build"
MODEL=""; INPUT=""; OUTPUT=""; INDEX=""; INDEX_RATE=""; PITCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --index) INDEX="$2"; shift 2;;
    --index-rate) INDEX_RATE="$2"; shift 2;;
    --pitch) PITCH="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --build) BUILD="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | tail -n +2 | sed 's/^# \?//'; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

[[ -n "$MODEL" && -n "$INPUT" && -n "$OUTPUT" ]] || {
  echo "error: --model, --input, --output are required" >&2; exit 1; }

HUBERT="$MODELS/hubert_base/model.safetensors"
RMVPE="$MODELS/rmvpe.safetensors"

ARGS=(--hubert "$HUBERT" --model "$MODEL" --rmvpe "$RMVPE"
      --input "$INPUT" --output "$OUTPUT"
      --version v2 --speakers 109 --sr 40000 --pitch "$PITCH")
if [[ -n "$INDEX" ]]; then
  ARGS+=(--index "$INDEX" --index-rate "${INDEX_RATE:-0.5}")
fi

"$BUILD/vc_convert" "${ARGS[@]}"
