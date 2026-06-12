#!/usr/bin/env bash
set -euo pipefail

# Run a complete speed + WER experiment for a given code change.
# Usage: bench/run_experiment.sh <label> [speed-runs]
# Requires: release binary at target/release/qwen-asr
# Outputs: bench/results/<label>/summary.json and librispeech-wer-bench/wer-results/<label>/summary.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LABEL="${1:-}"
RUNS="${2:-10}"

if [[ -z "$LABEL" ]]; then
    echo "Usage: $0 <label> [speed-runs]" >&2
    exit 1
fi

BINARY="$PROJECT_DIR/target/release/qwen-asr"
MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
SAMPLES_DIR="$PROJECT_DIR/bench/samples"
DATASET_DIR="$PROJECT_DIR/librispeech-wer-bench/dev-clean-2"
WER_OUTPUT="$PROJECT_DIR/librispeech-wer-bench/wer-results"
SPEED_OUTPUT="$PROJECT_DIR/bench/results"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found: $BINARY" >&2
    exit 1
fi

echo "============================================================"
echo "Experiment: $LABEL"
echo "Binary: $BINARY"
echo "============================================================"

echo ""
echo "--- Speed benchmark ($RUNS runs) ---"
bash "$SCRIPT_DIR/run.sh" \
    --binary "$BINARY" \
    --model-dir "$MODEL_DIR" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$SPEED_OUTPUT" \
    --label "$LABEL" \
    --runs "$RUNS"

echo ""
echo "--- WER benchmark (100 files, offline) ---"
python3 "$PROJECT_DIR/librispeech-wer-bench/librispeech_wer.py" \
    --limit 100 \
    --mode offline \
    --binary "$BINARY" \
    --model-dir "$MODEL_DIR" \
    --dataset "$DATASET_DIR" \
    --output-dir "$WER_OUTPUT" \
    --label "$LABEL"

echo ""
echo "============================================================"
echo "Experiment complete: $LABEL"
echo "Speed: $SPEED_OUTPUT/$LABEL/summary.json"
echo "WER:   $WER_OUTPUT/$LABEL/summary.json"
echo "============================================================"
