#!/bin/zsh
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "Usage: zsh $0 <input.wav> [python_executable]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  zsh $0 data/class1/class1_800Hz.wav" >&2
    echo "  PYTHON=/path/to/venv/bin/python zsh $0 data/class4/class4_2000Hz.wav" >&2
    exit 2
fi

wav_path="${1:A}"
python_bin="${PYTHON:-python3}"

if (( $# == 2 )); then
    python_bin="$2"
fi

if [[ ! -f "$wav_path" ]]; then
    echo "ERROR: WAV file not found: $wav_path" >&2
    exit 1
fi

script_dir="${0:A:h}"
cd "$script_dir"

"$python_bin" wav_to_mfcc_mem.py "$wav_path" \
    -o mem/mfcc_frames.mem \
    --frames-file mem/num_frames.mem

frames="$(tr -d '[:space:]' < mem/num_frames.mem)"
if [[ -z "$frames" || "$frames" != <-> ]]; then
    echo "ERROR: invalid frame count in mem/num_frames.mem: '$frames'" >&2
    exit 1
fi

iverilog -g2012 \
    -P tb_tm_pipeline.MAX_FRAMES="$frames" \
    -o sim \
    tb_tm_pipeline.v tm_pipeline.v mfcc_frame_source.v \
    binarizer.v tm_classifier.v

vvp sim
