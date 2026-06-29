#!/usr/bin/env python3

import os
import sys
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 15625
DURATION = 1.0
OUTPUT_DIR = "."

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <frequency_in_Hz>")
    sys.exit(1)

try:
    freq = float(sys.argv[1])
except ValueError:
    print("ERROR: Frequency must be a number.")
    sys.exit(1)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time vector
t = np.linspace(
    0,
    DURATION,
    int(SAMPLE_RATE * DURATION),
    endpoint=False
)

# Generate sine wave
signal = 0.8 * np.sin(2 * np.pi * freq * t)

# Convert to 16-bit PCM
audio = np.int16(signal * 32767)

# Output filename
if freq.is_integer():
    freq_str = f"{int(freq)}Hz"
else:
    freq_str = f"{freq:.2f}Hz".replace(".", "_")

filename = os.path.join(
    OUTPUT_DIR,
    f"sine_{freq_str}.wav"
)

write(filename, SAMPLE_RATE, audio)

print(f"Generated: {filename}")
