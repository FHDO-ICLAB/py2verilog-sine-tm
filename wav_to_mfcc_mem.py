#!/usr/bin/env python3
"""
wav_to_mfcc_mem.py

Convert a WAV file into Q8.8 MFCC values for Verilog simulation.

Output format:
  mem/mfcc_frames.mem contains one 16-bit hexadecimal Q8.8 value per line.

Layout:
  frame0_coeff0
  frame0_coeff1
  ...
  frame0_coeff12
  frame1_coeff0
  ...
"""

import argparse
import os
import numpy as np
import librosa

SAMPLE_RATE = 15625
FRAME_SIZE  = 400
HOP_SIZE    = 160
N_MEL       = 26
N_MFCC      = 13


def float_to_q88_hex(value: float) -> str:
    """Convert floating-point value to signed 16-bit Q8.8 hex."""
    q = int(round(value * 256.0))
    q = max(-32768, min(32767, q))

    if q < 0:
        q = (1 << 16) + q

    return f"{q & 0xFFFF:04X}"


def extract_mfcc_frames(wav_path: str) -> np.ndarray:
    """Extract MFCC frames using the same parameters as the training script."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    if len(y) < 2:
        raise ValueError("Input audio is too short.")

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=HOP_SIZE,
        win_length=FRAME_SIZE,
        window="hamming",
        n_mels=N_MEL,
        fmin=300,
        fmax=7500,
        power=2.0,
        center=False,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)

    return mfcc.T  # shape: frames x 13


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WAV audio to Q8.8 MFCC memory file for Verilog."
    )
    parser.add_argument("input_wav", help="Input WAV file")
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join("mem", "mfcc_frames.mem"),
        help="Output memory file (default: mem/mfcc_frames.mem)",
    )
    parser.add_argument(
        "--frames-file",
        default=os.path.join("mem", "num_frames.mem"),
        help="Optional output file containing the number of frames",
    )

    args = parser.parse_args()

    mfcc_frames = extract_mfcc_frames(args.input_wav)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames_dir = os.path.dirname(args.frames_file)
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    with open(args.output, "w") as f:
        for frame in mfcc_frames:
            for coeff in frame:
                f.write(float_to_q88_hex(float(coeff)) + "\n")

    with open(args.frames_file, "w") as f:
        f.write(str(len(mfcc_frames)) + "\n")

    print(f"Generated {len(mfcc_frames)} MFCC frames")
    print(f"Wrote MFCC data to: {args.output}")
    print(f"Wrote frame count to: {args.frames_file}")


if __name__ == "__main__":
    main()
