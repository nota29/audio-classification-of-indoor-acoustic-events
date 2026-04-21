"""
config.py — Central configuration for paths and hyperparameters.

No editing needed — all paths are relative to the project root.
Just place your data inside the data/ folder and outputs will go into outputs/.

Files you must create manually before running any script:
    data/meta.csv       — one Label per training audio file, same order as Datasets/Train/
    data/test_meta.csv  — one Label per test audio file, same order as Datasets/Test/
    See README.md > Data Setup for the expected format.

Expected data layout:
    data/
    ├── meta.csv                  ← create manually (training labels)
    ├── test_meta.csv             ← create manually (test labels)
    ├── Datasets/
    │   ├── Train/      ← create manually, place raw audio files here
    │   └── Test/       ← create manually, place raw audio files here
    ├── csvs/                     ← generated automatically by csv_creator.py
    │   ├── Train/
    │   │   ├── spec/   ← normal.csv, time_mask.csv, freq_mask.csv, time_warp.csv
    │   │   └── MFCC/   ← normal.csv, time_mask.csv, freq_mask.csv, time_warp.csv
    │   └── Test/
    │       ├── spec.csv
    │       └── MFCC.csv
    ├── spec_augments/Stereo64/   ← generated automatically by specaugment.py
    └── MFCC/Stereo64/            ← generated automatically by mfccaug.py
"""

import os

# ── Project root (folder where this file lives) ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Root directories (derived automatically — no editing needed) ──────────────
BASE_DATA_DIR   = os.path.join(BASE_DIR, 'data')
BASE_MODELS_DIR = os.path.join(BASE_DIR, 'outputs')

# ── Metadata CSVs (create these manually — see README > Data Setup) ──────────
# One row per audio sample with a 'Label' column, in the same order
# as the files in Datasets/Train/ and Datasets/Test/ respectively.
META_PATH      = os.path.join(BASE_DATA_DIR, 'meta.csv')
TEST_META_PATH = os.path.join(BASE_DATA_DIR, 'test_meta.csv')

# ── Dataset paths ─────────────────────────────────────────────────────────────
TRAIN_AUDIO_DIR = os.path.join(BASE_DATA_DIR, 'Datasets', 'Train')
TEST_AUDIO_DIR  = os.path.join(BASE_DATA_DIR, 'Datasets', 'Test')

# ── CSV paths ─────────────────────────────────────────────────────────────────
_CSV = os.path.join(BASE_DATA_DIR, 'csvs')

TRAIN_SPEC_NORMAL   = os.path.join(_CSV, 'Train', 'spec', 'normal.csv')
TRAIN_SPEC_TIMEMASK = os.path.join(_CSV, 'Train', 'spec', 'time_mask.csv')
TRAIN_SPEC_FREQMASK = os.path.join(_CSV, 'Train', 'spec', 'freq_mask.csv')
TRAIN_SPEC_TIMEWARP = os.path.join(_CSV, 'Train', 'spec', 'time_warp.csv')
TEST_SPEC           = os.path.join(_CSV, 'Test', 'spec.csv')

TRAIN_MFCC_NORMAL   = os.path.join(_CSV, 'Train', 'MFCC', 'normal.csv')
TRAIN_MFCC_TIMEMASK = os.path.join(_CSV, 'Train', 'MFCC', 'time_mask.csv')
TRAIN_MFCC_FREQMASK = os.path.join(_CSV, 'Train', 'MFCC', 'freq_mask.csv')
TRAIN_MFCC_TIMEWARP = os.path.join(_CSV, 'Train', 'MFCC', 'time_warp.csv')
TEST_MFCC           = os.path.join(_CSV, 'Test', 'MFCC.csv')

# ── Augmented image save paths ────────────────────────────────────────────────
SPEC_SAVE_DIR = os.path.join(BASE_DATA_DIR, 'spec_augments', 'Stereo64')
MFCC_SAVE_DIR = os.path.join(BASE_DATA_DIR, 'MFCC', 'Stereo64')

# ── Model output paths ────────────────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(BASE_MODELS_DIR, 'weights')
MODELS_DIR  = os.path.join(BASE_MODELS_DIR, 'saved')
METRICS_DIR = os.path.join(BASE_MODELS_DIR, 'metrics')

# ── Audio parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 64
N_CLASSES   = 9

# ── Training hyperparameters ──────────────────────────────────────────────────
LEARNING_RATE = 0.0004
VAL_SPLIT     = 0.25
EPOCHS        = 50
