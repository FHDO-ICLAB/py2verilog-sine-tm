#!/usr/bin/env python3
"""
train_tm.py  --  Train Tsetlin Machine on insect audio data, export to FPGA.

Workflow:
  1. Load .wav files from  data/<class_name>/*.wav
  2. Extract MFCC features (matching FPGA pipeline exactly)
  3. Binarize with thermometer encoding
  4. Train a multi-class Tsetlin Machine
  5. Export:  mem/tm_states.mem
              mem/binarizer_thresh.mem

Dependencies:
    pip install numpy scipy librosa tmu

"""

import numpy as np
import os
import sys
import glob
import struct

try:
    import librosa
except ImportError:
    print('ERROR: librosa not found.  pip install librosa')
    sys.exit(1)

try:
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        from tmu.tsetlin_machine import TMClassifier
except ImportError:
    print('ERROR: tmu not found.  pip install tmu')
    sys.exit(1)

# =============================================================================
# Configuration  (must match FPGA parameters)
# =============================================================================
SAMPLE_RATE  = 15625    # not 16000 - fixed because we are using 80 Mhz ...
FRAME_SIZE   = 400      # samples
HOP_SIZE     = 160      # samples
N_MEL        = 26
N_MFCC       = 13
N_THRESH     = 8        # thresholds per MFCC coefficient
N_FEATURES   = N_MFCC * N_THRESH  # = 104
N_LITERALS   = N_FEATURES * 2     # = 208

N_CLASSES    = 5
N_CLAUSES    = 20
T            = 15
S            = 2.0      # specificity
N_EPOCHS     = 10

# DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
# OUT_DIR      = os.path.join(os.path.dirname(__file__), '..', 'rtl')

DATA_DIR = os.path.abspath("./data")
OUT_DIR  = os.path.abspath("./mem")

CLASS_NAMES  = ['class0', 'class1', 'class2', 'class3', 'class4']

# =============================================================================
# MFCC extraction (matching FPGA pipeline)
# =============================================================================
def extract_mfcc_frames(wav_path):
    """Load audio and extract MFCC feature vectors."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # STFT -> Mel -> Log -> DCT
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=HOP_SIZE,
        win_length=FRAME_SIZE,
        window='hamming',
        n_mels=N_MEL,
        fmin=300,
        fmax=7500,
        power=2.0,
        center=False # ChatGPT Addition
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc  = librosa.feature.mfcc(S=log_S, n_mfcc=N_MFCC)

    # Transpose: each column is one frame  -> shape (n_frames, 13)
    return mfcc.T   # (n_frames, 13)


# =============================================================================
# Binarizer
# =============================================================================
def fit_thresholds(X_all):
    """
    Fit N_THRESH thresholds per coefficient using percentile spacing.
    X_all: (N_samples, N_MFCC)
    Returns: thresholds (N_MFCC, N_THRESH)
    """
    percentiles = np.linspace(100.0 / (N_THRESH + 1),
                              100.0 * N_THRESH / (N_THRESH + 1),
                              N_THRESH)
    thresholds = np.percentile(X_all, percentiles, axis=0).T  # (N_MFCC, N_THRESH)
    return thresholds


def quantize_q88(X):
    """Quantize MFCC values to signed Q8.8 integers, matching the RTL input."""
    return np.clip(np.round(X * 256), -32768, 32767).astype(np.int32)


def binarize(X, thresholds):
    """
    Thermometer encoding.
    X:          (N_samples, N_MFCC)
    thresholds: (N_MFCC, N_THRESH)
    Returns:    (N_samples, N_FEATURES)
    """
    N = X.shape[0]
    B = np.zeros((N, N_FEATURES), dtype=np.uint32)          # <-- uint8 -> uint32
    for c in range(N_MFCC):
        for t in range(N_THRESH):
            B[:, c * N_THRESH + t] = (X[:, c] > thresholds[c, t]).astype(np.uint32)  # <-- uint8 -> uint32
    return B


# =============================================================================
# Data loading
# =============================================================================
def load_dataset():
    X_list, y_list = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        pattern = os.path.join(DATA_DIR, cls_name, '*.wav')
        files = sorted(glob.glob(pattern))
        if not files:
            print(f'  WARNING: no .wav files found in data/{cls_name}/')
            continue
        for fpath in files:
            try:
                mfcc_frames = extract_mfcc_frames(fpath)
                X_list.append(mfcc_frames)
                y_list.extend([cls_idx] * len(mfcc_frames))
                print(f'  {cls_name}: {fpath} -> {len(mfcc_frames)} frames')
            except Exception as e:
                print(f'  ERROR loading {fpath}: {e}')

    if not X_list:
        print('\nERROR: No data found. Place .wav files in data/<class_name>/*.wav')
        sys.exit(1)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.uint32)
    return X, y


# =============================================================================
# Export helpers
# =============================================================================
def write_hex(filename, values, width_bits):
    hex_digits = (width_bits + 3) // 4
    mask = (1 << width_bits) - 1
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w') as f:
        for v in values:
            f.write(f'{int(v) & mask:0{hex_digits}X}\n')
    print(f'  Wrote {path}  ({len(values)} entries)')


def export_thresholds(thresholds):
    """Write Q8.8 thresholds to .mem file."""
    flat = thresholds.flatten()  # (N_MFCC * N_THRESH,)
    q88  = np.clip(np.round(flat), -32768, 32767).astype(np.int32)
    write_hex('binarizer_thresh.mem', q88, 16)

# =============================================================================
# Export model
# =============================================================================

def export_tm_states(model):
    states = []

    assert len(model.clause_banks) == N_CLASSES, \
        f"Expected {N_CLASSES} clause banks, got {len(model.clause_banks)}"

    for cls in range(N_CLASSES):
        cb = model.clause_banks[cls]
        assert cb.number_of_clauses == N_CLAUSES, \
            f"class {cls}: expected {N_CLAUSES} clauses, got {cb.number_of_clauses}"

        for clause in range(N_CLAUSES):
            for literal in range(N_LITERALS):
                states.append(cb.get_ta_state(clause, literal))

    flat = np.array(states, dtype=np.int32)

    expected = N_CLASSES * N_CLAUSES * N_LITERALS
    assert flat.size == expected, f"Expected {expected}, got {flat.size}"

    print("TM state export:")
    print("  flat size:", flat.size)
    print("  min/max  :", flat.min(), flat.max())

    assert flat.min() >= 0 and flat.max() <= 15, \
        f"TA states out of 4-bit range: min={flat.min()} max={flat.max()}"

    write_hex("tm_states.mem", flat, 4)

# =============================================================================
# Main
# =============================================================================
def main():
    print('=== Insect Classifier -- Tsetlin Machine Training ===\n')
    np.random.seed(1)

    # Load data
    print('Loading dataset ...')
    X, y = load_dataset()
    print(f'  Total frames: {len(X)}, Classes: {np.unique(y)}')

    # Match the RTL path: MFCC values are represented as signed Q8.8.
    X = quantize_q88(X)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train / test split  80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit binarizer thresholds on training data
    print('\nFitting binarizer thresholds ...')
    thresholds = np.clip(
        np.round(fit_thresholds(X_train)),
        -32768,
        32767
    ).astype(np.int32)

    # Binarize
    print('Binarizing features ...')
    X_train_bin = binarize(X_train, thresholds)
    X_test_bin  = binarize(X_test,  thresholds)

    # Train Tsetlin Machine
    print(f'\nTraining TM ({N_EPOCHS} epochs, {N_CLAUSES} clauses/class, T={T}) ...')
    tm = TMClassifier(
         number_of_clauses=N_CLAUSES,# * N_CLASSES, # UNTERSCHIED MIT 82% oder OHNE 72% bei 1 Epoche
        #number_of_clauses=N_CLAUSES, # ChatGPT
        T=T,
        s=S,
        platform='CPU',
        number_of_state_bits_ta=4, # ChatGPT ADDED !!!
        incremental=False
    )
    for epoch in range(N_EPOCHS):
        tm.fit(X_train_bin, y_train)
        if (epoch + 1) % 10 == 0:
            acc = np.mean(tm.predict(X_test_bin) == y_test) * 100
            print(f'  Epoch {epoch+1:3d}: test accuracy = {acc:.1f}%')

    acc_final = np.mean(tm.predict(X_test_bin) == y_test) * 100
    print(f'\nFinal test accuracy: {acc_final:.1f}%')

    # Export
    print('\nExporting to FPGA ROM files ...')
    os.makedirs(OUT_DIR, exist_ok=True)
    export_thresholds(thresholds)

    # Debugging ...
    #print("Checking tm object ...")
    #print(type(tm))
    #print([x for x in dir(tm) if "state" in x.lower() or "ta" in x.lower() or "clause" in x.lower()])
    #ta = tm.get_ta_state()
    #print(type(ta))
    #print(len(ta))

    export_tm_states(tm)

    print('\nDone.')


if __name__ == '__main__':
    main()
