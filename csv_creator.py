"""
csv_creator.py — Creates CSV files mapping image filepaths to their labels.

Run this script after generating your augmented images (specaugment.py / mfccaug.py)
and before training. It scans each image folder and pairs each image path with its
corresponding label from the metadata CSV.

Usage:
    Set flag to 0 for mel-spectrograms or 1 for MFCCs, then run:
        python csv_creator.py

The metadata CSV (meta.csv) must exist in your data/ folder and contain a 'Label' column
with one label per audio sample, in the same order as the audio files.
"""

import os
import re
import csv
import pandas as pd

from config import (
    META_PATH, TEST_META_PATH,
    BASE_DATA_DIR,
    SPEC_SAVE_DIR, MFCC_SAVE_DIR,
    TRAIN_SPEC_NORMAL, TRAIN_SPEC_TIMEMASK, TRAIN_SPEC_FREQMASK, TRAIN_SPEC_TIMEWARP,
    TRAIN_MFCC_NORMAL, TRAIN_MFCC_TIMEMASK, TRAIN_MFCC_FREQMASK, TRAIN_MFCC_TIMEWARP,
    TEST_SPEC, TEST_MFCC,
)


# ── Config ────────────────────────────────────────────────────────────────────
# Set to 0 for mel-spectrograms, 1 for MFCCs
FLAG = 1

# ── Utilities ─────────────────────────────────────────────────────────────────

def sorted_natural(data):
    """Sort a list naturally (image2 before image10) instead of alphabetically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def csv_creator(filename, fields, rows):
    """
    Save a CSV file with the given fields (header) and rows (data).

    Args:
        filename : Full path to save the CSV
        fields   : List of column names
        rows     : List of [filepath, label] pairs
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"Saved: {filename}  ({len(rows)} rows)")


def create_csv(image_dir, labels, save_path):
    """
    Scan image_dir, pair each image with its label, and save as a CSV.

    Args:
        image_dir : Folder containing the generated images
        labels    : Pandas Series of labels (from metadata CSV)
        save_path : Where to save the resulting CSV
    """
    paths = sorted_natural(os.listdir(image_dir))
    rows  = [[os.path.join(image_dir, p), labels[i]] for i, p in enumerate(paths)]
    csv_creator(save_path, ['Filepath', 'Label'], rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    meta   = pd.read_csv(META_PATH)
    labels = meta['Label']
    print(f"Loaded metadata: {len(labels)} samples")

    if FLAG == 0:
        print("\nGenerating mel-spectrogram CSVs...")
        create_csv(os.path.join(SPEC_SAVE_DIR, 'Train'),    labels, TRAIN_SPEC_NORMAL)
        create_csv(os.path.join(SPEC_SAVE_DIR, 'Timemask'), labels, TRAIN_SPEC_TIMEMASK)
        create_csv(os.path.join(SPEC_SAVE_DIR, 'Freqmask'), labels, TRAIN_SPEC_FREQMASK)
        create_csv(os.path.join(SPEC_SAVE_DIR, 'Timewarp'), labels, TRAIN_SPEC_TIMEWARP)

        # Test set — load test metadata separately
        test_meta   = pd.read_csv(TEST_META_PATH)
        test_labels = test_meta['Label']
        create_csv(os.path.join(SPEC_SAVE_DIR, 'Test'), test_labels, TEST_SPEC)

    elif FLAG == 1:
        print("\nGenerating MFCC CSVs...")
        create_csv(os.path.join(MFCC_SAVE_DIR, 'Train'),    labels, TRAIN_MFCC_NORMAL)
        create_csv(os.path.join(MFCC_SAVE_DIR, 'Timemask'), labels, TRAIN_MFCC_TIMEMASK)
        create_csv(os.path.join(MFCC_SAVE_DIR, 'Freqmask'), labels, TRAIN_MFCC_FREQMASK)
        create_csv(os.path.join(MFCC_SAVE_DIR, 'Timewarp'), labels, TRAIN_MFCC_TIMEWARP)

        # Test set — load test metadata separately
        test_meta   = pd.read_csv(TEST_META_PATH)
        test_labels = test_meta['Label']
        create_csv(os.path.join(MFCC_SAVE_DIR, 'Test'), test_labels, TEST_MFCC)

    print("\nDone — all CSVs created.")
