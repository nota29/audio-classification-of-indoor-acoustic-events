# Audio Classification via Spectrogram Imaging

A deep learning pipeline for audio classification of domestic activities using image-based representations of audio signals. The domestic activities are: absence, cooking, dishwashing, eating, social activity,vacuum cleaning, watching TV, working, other(every other indoor activity). Raw audio is converted into **mel-spectrograms** or **MFCC images** with SpecAugment-style data augmentation, then classified using seven custom-built CNN architectures implemented from scratch in TensorFlow/Keras.

---

## Overview

The pipeline has three main stages:

1. **Feature Extraction & Augmentation** — Audio signals are transformed into 2D image representations (mel-spectrograms or MFCCs) using four augmentation strategies: normal, time masking, frequency masking, and time warping.
2. **Model Training** — A single unified training script trains any of the seven CNN architectures, using class-weight balancing, early stopping, and learning rate scheduling.
3. **Evaluation** — Each model is evaluated on a held-out test set with accuracy, loss, F1 score, and confusion matrix plots.

---

## Repository Structure

```
audio-classification/
│
├── augmentation/
│   ├── specaugment.py      # Mel-spectrogram extraction + SpecAugment augmentation
│   └── mfccaug.py          # MFCC image extraction + SpecAugment augmentation
│
├── models/
│   └── create_model.py     # All 7 model architectures (factory functions)
│
├── utils/                  # Dataloader, metrics, logger, trainer helpers
│   ├── dataloader_.py      
│   ├── metrics_.py 
│   ├── mylogger_.py
│   └── train_model_.py
│
├── train.py                # Unified training script (all models, one file)
├── csv_creator.py          # Creates image filepath/label CSVs for training
├── config.py               # All paths and hyperparameters — edit this first
├── requirements.txt
└── README.md
```

---

## Models

Conv1D and Conv2D architectures (with both uniform and asymmetric kernels) are implemented from scratch in TensorFlow/Keras. The remaining models are custom implementations inspired by the original papers listed below. Each model outputs probabilities over **9 classes** via a softmax layer.


| Model | `--model` arg | Input | Key Design |
|---|---|---|---|
| **Conv1D** | `conv1d` | MFCC images | Group norm, weight norm, GELU, L2 reg |
| **Conv2D Same Kernels** | `conv2d_same` | Mel-spectrograms | Symmetric (3×3, 5×5) kernels, batch norm |
| **Conv2D Diff Kernels** | `conv2d_diff` | Mel-spectrograms | Asymmetric (5×1, 10×1, 1×6) kernels — frequency-aware |
| **Custom MobileNet V1** | `mobile_v1` | Mel-spectrograms | Depthwise separable convolutions, ReLU6 |
| **Custom MobileNet V3 Small** | `mobile_v3_small` | Mel-spectrograms | Squeeze-and-excitation, hard-swish, inverted residuals |
| **Custom MobileNet V3 Large** | `mobile_v3_large` | Mel-spectrograms | Wider V3 with 15 bottleneck stages |
| **Custom EfficientNet-B0** | `efficientnet` | Mel-spectrograms | MBConv blocks, Swish activation, Squeeze-and-excitation blocks |

---

## Dataset

This project uses the **SINS database (DCASE 2018, Task 5)** — a dataset for monitoring domestic activities based on multi-channel acoustics.

**Download:**
- Development set (training): available via the [DCASE 2018 Challenge page](https://zenodo.org/records/1247102)
- Evaluation set (test):https://zenodo.org/records/1964758

> Note: The dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Please cite the original authors when using it (see Citation section below).
---

## Data Augmentation

Audio signals are sampled at **16 kHz** and converted to grayscale images using:

- **Mel-spectrograms** (`specaugment.py`) — 64 mel bands, Hann window, hop length 512, n_fft 2048
- **MFCC images** (`mfccaug.py`) — 64 MFCC coefficients derived from mel-spectrograms

Each audio sample is converted four ways using **SpecAugment** (Park et al., 2019):

| Mode | Description |
|---|---|
| `normal` | No augmentation |
| `time_mask` | Random time-axis masking (param=10) |
| `freq_mask` | Random frequency-axis masking (param=10) |
| `time_warp` | Sparse image warp along time axis (policy LB: W=80) |

Images are min-max scaled to [0, 255], vertically flipped, and inverted before saving.

---

## Results

| Model | Test Accuracy | F1 Score | Params |
|---|---|---|---|
| Conv1D | 0.9150 | 0.8613 | ~3.3M |
| Conv2D Same Kernels | 0.9167| 0.8699 | ~950K |
| Conv2D Diff Kernels | 0.9033 | 0.8563 | ~300K |
| MobileNet V1 | 0.8709 | 0.8446 | ~3.2M |
| MobileNet V3 Small | 0.8726 | 0.8434 | ~1.7M |
| MobileNet V3 Large | 0.8843| 0.8507| ~4.3M |
| EfficientNet-B0 | 0.8974| 0.8608| ~7M |

---

## Training Details

All models share a common training setup (configured in `train.py` and `config.py`):

| Parameter | Value |
|---|---|
| Optimizer | Adam + AMSGrad |
| Learning rate | 0.0004 |
| LR schedule | ReduceLROnPlateau (patience=2, factor=0.92) |
| Early stopping | patience=8, restore best weights |
| Max epochs | 50 |
| Validation split | 25% |
| Loss | Categorical cross-entropy |
| Class imbalance | sklearn class weight balancing |
| Weight init | He normal |

Batch sizes vary by model: 64 (Conv1D, MobileNets), 32 (Conv2D Same), 16 (Conv2D Diff, EfficientNet).

---

## Installation

```bash
git clone https://github.com/your-username/audio-classification.git
cd audio-classification
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare your data folder

Set up your `data/` folder according to the layout described in `config.py`.

### 2. Generate augmented image datasets

```bash
# Mel-spectrogram images
python augmentation/specaugment.py

# MFCC images
python augmentation/mfccaug.py
```

### 3. Create the CSV files

```bash
python csv_creator.py
```

Set `FLAG = 0` for mel-spectrograms or `FLAG = 1` for MFCCs at the top of the script. This scans your generated image folders and creates the filepath/label CSVs needed for training.

### 4. Train a model

```bash
# Train any model by name
python train.py --model mobile_v3_large

# Optionally give the run a custom name
python train.py --model mobile_v3_large --run_name experiment_v2
```

The script automatically selects the correct CSVs, batch size, and model architecture. It saves weights, the trained model, confusion matrix, metric plots, and logs results.

### Available model names

```
conv1d  |  conv2d_same  |  conv2d_diff  |  mobile_v1  |  mobile_v3_small  |  mobile_v3_large  |  efficientnet
```

---

## Dependencies

See `requirements.txt`. Key libraries:

- TensorFlow ≥ 2.10 + TensorFlow Addons + TensorFlow IO
- Librosa (audio loading and feature extraction)
- scikit-image (image saving)
- scikit-learn (class weights, metrics)
- NumPy, Pandas, Matplotlib

---


## References

- [MobileNetV1](https://arxiv.org/abs/1704.04861)   
- [MobileNetV3](https://arxiv.org/abs/1905.02244)  
- [EfficientNet](https://arxiv.org/abs/1905.11946)

---

## Citation

If you use this work, please cite:

```
@misc{audio-classification,
  author = {nota},
  title  = {Indoor acoustic events classification via Spectrogram Imaging},
  year   = {2026},
  url    = {https://github.com/your-username/audio-classification-of-indoor-acoustic-events}
}
```

If you use the SINS dataset, please cite:

```
@inproceedings{Dekkers2017,
  author    = {Dekkers, G. and Lauwereins, S. and Thoen, B. and Adhana, M. and Brouckxon, H. and
               Van den Bergh, B. and van Waterschoot, T. and Vanrumste, B. and Verhelst, M. and Karsmakers, P.},
  title     = {The SINS database for detection of daily activities in a home environment using an Acoustic Sensor Network},
  booktitle = {Proceedings of the DCASE 2017 Workshop},
  address   = {Munich, Germany},
  year      = {2017},
  pages     = {32--36}
}
