"""
mfccaug.py — MFCC feature extraction with SpecAugment augmentation.

Converts raw audio files into grayscale MFCC images using four modes:
    normal     : no augmentation
    time_mask  : random time-axis masking applied before MFCC extraction
    freq_mask  : random frequency-axis masking applied before MFCC extraction
    time_warp  : sparse image warp along the time axis (SpecAugment, Park et al. 2019)

Audio parameters (set in config.py): SAMPLE_RATE=16000, N_MELS=64
"""

import os
import re
import random
import numpy as np
import librosa
import tensorflow_io as tfio
from skimage.io import imsave
from tensorflow_addons.image import sparse_image_warp

from config import SAMPLE_RATE, N_MELS, TRAIN_AUDIO_DIR, TEST_AUDIO_DIR, MFCC_SAVE_DIR


# ── Utilities ─────────────────────────────────────────────────────────────────

def sorted_natural(data):
    """Sort a list naturally (image2 before image10) instead of alphabetically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def scale_minmax(X, minimum, maximum):
    """Rescale array X to the range [minimum, maximum]."""
    x_std = (X - X.min()) / (X.max() - X.min())
    return x_std * (maximum - minimum)


def time_warp(spec, policy='LB'):
    """
    Apply time warping to a spectrogram (SpecAugment, Park et al. 2019).

    Args:
        spec   : 2D spectrogram array (freq x time)
        policy : SpecAugment policy — 'LB', 'LD', 'SM', or 'SS'

    Returns:
        Time-warped spectrogram tensor.
    """
    W = {'LB': 80, 'LD': 80, 'SM': 40, 'SS': 40}[policy]
    v, tau = spec.shape[0], spec.shape[1]

    random_pt = spec[v // 2][random.randrange(W, tau - W)]
    w = np.random.uniform(-W, W)

    warped, _ = sparse_image_warp(
        spec,
        src_control_point_locations=[[[v // 2, random_pt]]],
        dest_control_point_locations=[[[v // 2, random_pt + w]]],
        num_boundary_points=2,
    )
    return warped


# ── Core function ─────────────────────────────────────────────────────────────

def mfcc_augment(init_path, sr, mode, savepath, n_mels):
    """
    Convert all audio files in init_path to MFCC images and save them.

    MFCC extraction flow:
        audio → mel-spectrogram (power_to_db) → [augment] → MFCC → image

    Args:
        init_path : Directory of raw audio files
        sr        : Sample rate for loading audio
        mode      : One of 'normal', 'time_mask', 'freq_mask', 'time_warp'
        savepath  : Save path template with {} placeholder for the image index
        n_mels    : Number of MFCC coefficients
    """
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    samples = sorted_natural(os.listdir(init_path))

    for count, sample in enumerate(samples):
        y, _ = librosa.load(os.path.join(init_path, sample),
                            sr=sr, res_type='kaiser_fast', mono=True)

        mel  = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        s_db = librosa.power_to_db(abs(mel), ref=np.max)

        # Apply augmentation before MFCC extraction
        if mode == 'time_mask':
            s_db = tfio.audio.time_mask(s_db, param=10).numpy()
        elif mode == 'freq_mask':
            s_db = tfio.audio.freq_mask(s_db, param=10).numpy()
        elif mode == 'time_warp':
            s_db = time_warp(s_db).numpy()
        # 'normal' needs no modification

        mfcc  = librosa.feature.mfcc(S=s_db, sr=sr, n_mfcc=n_mels)
        image = scale_minmax(mfcc, 0, 255).astype(np.uint8)
        image = 255 - np.flip(image, axis=0)

        imsave(savepath.format(count), image)
        print(f"[mfcc/{mode}] Saved #{count}")

    print(f"Done — {len(samples)} images saved to {os.path.dirname(savepath)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    Train set 
    mfcc_augment(TRAIN_AUDIO_DIR, SAMPLE_RATE, 'normal',    os.path.join(MFCC_SAVE_DIR, 'Train',    'image{}.png'), N_MELS)
    mfcc_augment(TRAIN_AUDIO_DIR, SAMPLE_RATE, 'time_mask', os.path.join(MFCC_SAVE_DIR, 'Timemask', 'image{}.png'), N_MELS)
    mfcc_augment(TRAIN_AUDIO_DIR, SAMPLE_RATE, 'freq_mask', os.path.join(MFCC_SAVE_DIR, 'Freqmask', 'image{}.png'), N_MELS)
    mfcc_augment(TRAIN_AUDIO_DIR, SAMPLE_RATE, 'time_warp', os.path.join(MFCC_SAVE_DIR, 'Timewarp', 'image{}.png'), N_MELS)

    # Test set (no augmentation)
    mfcc_augment(TEST_AUDIO_DIR, SAMPLE_RATE, 'normal',
                 os.path.join(MFCC_SAVE_DIR, 'Test', 'image{}.png'), N_MELS)

    print("End of script")
