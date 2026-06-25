#!/usr/bin/env python3

import os
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 15625
DURATION = 1.0

CLASSES = {
    "class0": 400,
    "class1": 800,
    "class2": 1200,
    "class3": 1600,
    "class4": 2000,
}

BASE_DIR = "data"

# Create time vector
t = np.linspace(
    0,
    DURATION,
    int(SAMPLE_RATE * DURATION),
    endpoint=False
)

os.makedirs(BASE_DIR, exist_ok=True)

for class_name, freq in CLASSES.items():

    class_dir = os.path.join(BASE_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    signal = 0.8 * np.sin(2 * np.pi * freq * t)

    audio = np.int16(signal * 32767)

    filename = os.path.join(
        class_dir,
        f"{class_name}_{freq}Hz.wav"
    )

    write(filename, SAMPLE_RATE, audio)

    print(f"Generated: {filename}")

print("\nDone.")
