"""
train.py — Unified training script for all 7 audio classification models.

Usage:
    python train.py --model <model_name> [--run_name <name>]

Available models:
    conv1d
    conv2d_same
    conv2d_diff
    mobile_v1
    mobile_v3_small
    mobile_v3_large
    efficientnet

Examples:
    python train.py --model mobile_v3_large
    python train.py --model conv1d --run_name experiment_1
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from config import (
    TRAIN_SPEC_NORMAL, TRAIN_SPEC_TIMEMASK, TRAIN_SPEC_FREQMASK, TRAIN_SPEC_TIMEWARP,
    TRAIN_MFCC_NORMAL, TRAIN_MFCC_TIMEMASK, TRAIN_MFCC_FREQMASK, TRAIN_MFCC_TIMEWARP,
    TEST_SPEC, TEST_MFCC,
    WEIGHTS_DIR, MODELS_DIR, METRICS_DIR,
    LEARNING_RATE, VAL_SPLIT, EPOCHS,
)
from models.create_model import (
    conv1d,
    conv2d_same_kernels,
    conv2d_diff_kernels,
    mobile_net_v1,
    mobile_net_v3_small,
    mobile_net_v3_large,
    eff_net,
)
from utils import dataloader_
from utils.mylogger_ import logger
from utils.metrics_ import confmatrixplot, metric_plot, report_f1score
from utils.train_model_ import training

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry defines: csv_mode, batch_size, and the factory function.
# csv_mode='mfcc' uses MFCC CSVs; csv_mode='spec' uses mel-spectrogram CSVs.

MODEL_REGISTRY = {
    'conv1d': {
        'csv_mode':  'mfcc',
        'batchsize': 64,
        'build': lambda name, x, y, init: conv1d(
            x, y, n_channels=1,
            grouping_num=16,
            initializer=init,
            regularizer=tf.keras.regularizers.L2(l2=0.001),
            name=name,
            activation='gelu',
        ),
    },
    'conv2d_same': {
        'csv_mode':  'spec',
        'batchsize': 32,
        'build': lambda name, x, y, init: conv2d_same_kernels(
            name, x, y, n_channels=1,
            initializer=init, reg=None, batchsize=32, logits=False,
        ),
    },
    'conv2d_diff': {
        'csv_mode':  'spec',
        'batchsize': 16,
        'build': lambda name, x, y, init: conv2d_diff_kernels(
            name, x, y, n_channels=1,
            initializer=init, reg=None, batchsize=16, logits=False,
        ),
    },
    'mobile_v1': {
        'csv_mode':  'spec',
        'batchsize': 64,
        'build': lambda name, x, y, init: mobile_net_v1(
            name, x, y, n_channels=1,
            initializer=init, reg=None, batchsize=64, logits=False,
        ),
    },
    'mobile_v3_small': {
        'csv_mode':  'spec',
        'batchsize': 64,
        'build': lambda name, x, y, init: mobile_net_v3_small(
            name, x, y, n_channels=1,
            initializer=init, reg=None, batchsize=64, logits=False,
        ),
    },
    'mobile_v3_large': {
        'csv_mode':  'spec',
        'batchsize': 64,
        'build': lambda name, x, y, init: mobile_net_v3_large(
            name, x, y, n_channels=1,
            initializer=init, reg=None, batchsize=64, logits=False,
        ),
    },
    'efficientnet': {
        'csv_mode':  'spec',
        'batchsize': 16,
        'build': lambda name, x, y, init: eff_net(
            name, x, y, n_channels=1,
            initializer=init, reg=None, batchsize=16, logits=False,
        ),
    },
}


def load_train_csvs(csv_mode):
    """Load and shuffle the four augmented training CSVs for the given mode."""
    if csv_mode == 'mfcc':
        paths = [TRAIN_MFCC_NORMAL, TRAIN_MFCC_TIMEMASK, TRAIN_MFCC_FREQMASK, TRAIN_MFCC_TIMEWARP]
    else:
        paths = [TRAIN_SPEC_NORMAL, TRAIN_SPEC_TIMEMASK, TRAIN_SPEC_FREQMASK, TRAIN_SPEC_TIMEWARP]

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.sample(frac=1).reset_index(drop=True)


def load_test_csv(csv_mode):
    """Load and shuffle the test CSV for the given mode."""
    path = TEST_MFCC if csv_mode == 'mfcc' else TEST_SPEC
    df = pd.read_csv(path)
    return df.sample(frac=1).reset_index(drop=True)


def get_image_dims(df):
    """Read one sample image to determine the input shape."""
    sample = plt.imread(df['Filepath'][0])
    return sample.shape[0], sample.shape[1]


def build_callbacks(model_name, weights_dir):
    """Return the list of Keras callbacks used during training."""
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, f'best_weights_{model_name}.hdf5')

    save_best = tf.keras.callbacks.ModelCheckpoint(
        weight_path, save_best_only=True, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=2, factor=0.92,
        min_lr=0, mode='min', cooldown=0, verbose=1,
    )
    return [save_best, early_stop, reduce_lr], early_stop


def run_training(model_name, run_name=None):
    """Full training pipeline for the chosen model."""

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    cfg       = MODEL_REGISTRY[model_name]
    batchsize = cfg['batchsize']
    csv_mode  = cfg['csv_mode']
    full_name = run_name if run_name else model_name

    print(f"\n{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Run name   : {full_name}")
    print(f"  CSV mode   : {csv_mode}")
    print(f"  Batch size : {batchsize}")
    print(f"{'='*60}\n")

    # ── Data loading ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_df = load_train_csvs(csv_mode)
    test_df  = load_test_csv(csv_mode)
    x_size, y_size = get_image_dims(train_df)

    datagen = ImageDataGenerator(validation_split=VAL_SPLIT)
    train_it, val_it, train_size, val_size = dataloader_.load_data(
        datagen, train_df=train_df, val_split=VAL_SPLIT,
        batchsize=batchsize, mode='training', test_df=None,
    )
    my_weights = dataloader_.class_weight_calc(train_df, class_mode='sklearn')

    # ── Model creation ────────────────────────────────────────────────────────
    print("Building model...")
    initializer = tf.keras.initializers.he_normal()
    model = cfg['build'](full_name, x_size, y_size, initializer)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    weights_dir = os.path.join(WEIGHTS_DIR, model_name)
    callback_list, early_stop_cb = build_callbacks(full_name, weights_dir)

    # ── Compile & train ───────────────────────────────────────────────────────
    opt = Adam(learning_rate=LEARNING_RATE, amsgrad=True)
    model, history = training(
        model, opt, 'categorical_crossentropy',
        callback_list, my_weights, batchsize,
        train_size, val_size, EPOCHS, train_it, val_it,
    )

    # ── Save model ────────────────────────────────────────────────────────────
    save_path = os.path.join(MODELS_DIR, model_name, full_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("Evaluating on test set...")
    test_datagen = ImageDataGenerator()
    test_it, test_size = dataloader_.load_data(
        test_datagen, train_df=None, test_df=test_df,
        val_split=None, batchsize=batchsize, mode='test',
    )
    score = model.evaluate(test_it, steps=-(-test_size // batchsize), verbose=1)
    print(f"Test loss    : {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")

    # ── Predictions & metrics ─────────────────────────────────────────────────
    print("Predicting...")
    y_pred = model.predict(
        test_it, steps=-(-test_size // batchsize),
        batch_size=batchsize, verbose=1,
    )
    y_pred = np.argmax(y_pred, axis=1)

    metrics_dir = os.path.join(METRICS_DIR, model_name)
    os.makedirs(metrics_dir, exist_ok=True)

    confmatrixplot(test_df, y_pred,
                   Filepath=os.path.join(metrics_dir, f'conf_{full_name}.png'))
    metric_plot(history,
                Filepath=os.path.join(metrics_dir, f'metrics_{full_name}.png'))
    f1score = report_f1score(test_df, y_pred)

    # ── Logging ───────────────────────────────────────────────────────────────
    best_epoch = early_stop_cb.best_epoch
    min_val_loss = round(early_stop_cb.best, 5)
    params = model.count_params()
    desc = (
        f'Adam_amsgrad_lr={LEARNING_RATE}_batchsize={batchsize}'
        f'_valsplit={VAL_SPLIT}_epochs={EPOCHS}_csvmode={csv_mode}'
    )

    print("Logging results...")
    logger(
        full_name, params,
        round(history.history['accuracy'][best_epoch], 4),
        round(history.history['loss'][best_epoch], 5),
        round(history.history['val_accuracy'][best_epoch], 4),
        min_val_loss,
        round(score[1], 4),
        round(score[0], 4),
        best_epoch,
        len(history.history['loss']),
        round(f1score, 4),
        desc,
    )
    print("Done.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an audio classification model.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help='Model architecture to train.',
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Optional custom name for this run (used in saved filenames).',
    )
    args = parser.parse_args()
    run_training(args.model, args.run_name)
