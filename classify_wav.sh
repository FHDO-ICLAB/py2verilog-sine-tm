#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: sh $0 <input.wav> [python_executable]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  sh $0 data/class1/class1_800Hz.wav" >&2
    echo "  PYTHON=/path/to/venv/bin/python sh $0 data/class4/class4_2000Hz.wav" >&2
    exit 2
fi

# Convert the input path to an absolute path
case "$1" in
    /*) wav_path="$1" ;;
    *)  wav_path="$(pwd)/$1" ;;
esac

python_bin="${PYTHON:-python3}"

if [ "$#" -eq 2 ]; then
    python_bin="$2"
fi

if [ ! -f "$wav_path" ]; then
    echo "ERROR: WAV file not found: $wav_path" >&2
    exit 1
fi

# Determine the directory containing this script
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir"

"$python_bin" wav_to_mfcc_mem.py "$wav_path" \
    -o mem/mfcc_frames.mem \
    --frames-file mem/num_frames.mem

frames=$(tr -d '[:space:]' < mem/num_frames.mem)

case "$frames" in
    ''|*[!0-9]*)
        echo "ERROR: Invalid frame count in mem/num_frames.mem: '$frames'" >&2
        exit 1
        ;;
esac

iverilog -g2012 \
    -P tb_tm_pipeline.MAX_FRAMES="$frames" \
    -o sim \
    tb_tm_pipeline.v tm_pipeline.v mfcc_frame_source.v \
    binarizer.v tm_classifier.v

vvp sim
