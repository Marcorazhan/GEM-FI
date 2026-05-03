# GEM-FI: Gated Evidential Mixtures with Fisher Modulation

This repository contains the official implementation of **GEM-FI: Gated Evidential Mixtures with Fisher Modulation**, accepted as a regular paper at **ICML 2026**.

GEM-FI is a single-pass uncertainty estimation framework for deep neural networks. It combines evidential deep learning with learned energy-based gating, mixture-based epistemic modeling, and Fisher Information (FI)-based stabilization. The method is designed for reliable uncertainty quantification, confidence calibration, and out-of-distribution (OOD) detection.

## Overview

The GEM family contains three main variants:

- **GEM-CORE**: A single-head gated evidential model with a learned energy-to-gate mechanism.
- **GEM-MIX**: Extends GEM-CORE with a Mixture of Beliefs (MoB) to capture multi-modal epistemic uncertainty in a single forward pass.
- **GEM-FI**: Extends GEM-MIX with Fisher Information regularization and FI-based modulation to stabilize mixture allocations and improve uncertainty calibration.

## Key Features

- **Single-pass inference**: No Monte Carlo sampling or multi-pass ensembling is required at test time.
- **Energy-to-gate evidential learning**: A learned energy signal is mapped to bounded class-wise gates that modulate predictive probabilities.
- **Mixture of evidential heads**: Multiple Dirichlet heads model epistemic multi-modality while sharing one backbone.
- **Fisher Information modulation**: FI-based regularization helps reduce unstable mixture allocations and head collapse.
- **OOD detection**: Supports OOD detection using aleatoric and epistemic uncertainty scores.
- **Virtual Outlier Synthesis (VOS)**: Optional synthetic outlier generation for improved OOD detection in the full GEM-FI pipeline.
- **Reproducible experiments**: Includes training commands and hyperparameters used in the paper.

## Repository Structure

```text
.
├── main.py                 # Main training and evaluation entry point
├── train.py                # Training loops and model optimization logic
├── utility.py              # Data loading, model loading, and helper functions
├── density_estimation.py   # GDA/GMM density estimation utilities
├── conf_calibration.py     # Calibration metrics such as ECE
├── load_corrupted.py       # Loaders for corrupted datasets
├── requirements.txt        # Python dependencies
├── saved_results/          # Saved experimental outputs
└── data/                   # Dataset directory
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Marcorazhan/GEM-FI.git
cd GEM-FI
```

### 2. Create an environment

Using `conda`:

```bash
conda create -n gem-fi python=3.11
conda activate gem-fi
```

Or using `venv`:

```bash
python3.11 -m venv gem-fi-env
source gem-fi-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The code was developed with **Python 3.11** and optimized for **CUDA 12.1** and **PyTorch 2.3.0**.

## Data Preparation

The following standard datasets are downloaded automatically when needed:

- CIFAR-10
- CIFAR-100
- MNIST
- SVHN
- FashionMNIST
- KMNIST

For OOD and robustness benchmarks, download the following datasets manually and extract them into the `data` directory:

### TinyImageNet

Download `tiny-imagenet-200.zip`, for example from the Stanford CS231n mirror, and extract it to:

```text
./data/tiny-imagenet-200
```

### CIFAR-10-C

Download CIFAR-10-C from Zenodo and extract it to:

```text
./data/CIFAR-10-C
```

### MNIST-C

Download MNIST-C from Zenodo and extract it to:

```text
./data/mnist_c
```

The expected directory structure is:

```text
data/
├── tiny-imagenet-200/
├── CIFAR-10-C/
└── mnist_c/
```

## Usage

The main entry point is `main.py`. Different GEM variants can be trained using command-line arguments.

### Train GEM-CORE on CIFAR-10

```bash
python main.py --ID_dataset CIFAR-10 --backbone ResNet18
```

### Train GEM-MIX on CIFAR-10

```bash
python main.py --ID_dataset CIFAR-10 --backbone ResNet18 \
  --use_mob --num_components 3
```

### Train GEM-FI on CIFAR-10

```bash
python main.py --ID_dataset CIFAR-10 --backbone ResNet18 --batch_size 128 \
  --num_epochs 100 --learning_rate 1e-3 --dropout_rate 0.1 --reg_param 1e-4 \
  --use_mob --num_components 3 --use_fi_regularization --fi_lambda 0.1
```

### Train GEM-FI on MNIST

```bash
python main.py --ID_dataset MNIST --backbone ConvNet3C3F --batch_size 64 \
  --num_epochs 50 --learning_rate 5e-4 --dropout_rate 0.05 --reg_param 1e-3 \
  --use_mob --num_components 3 --use_fi_regularization --fi_lambda 0.3
```

> **Note**
> These commands reflect the main hyperparameters used in the paper for reproducibility.

## Important Arguments

| Argument | Description |
|---|---|
| `--ID_dataset` | In-distribution dataset. Supported options include `CIFAR-10`, `CIFAR-100`, and `MNIST`. |
| `--backbone` | Backbone architecture, such as `ResNet18`, `VGG16`, or `ConvNet3C3F`. |
| `--batch_size` | Training batch size. |
| `--num_epochs` | Number of training epochs. |
| `--learning_rate` | Initial learning rate. |
| `--dropout_rate` | Dropout rate used in the model. |
| `--reg_param` | Regularization parameter. |
| `--use_mob` | Enables the Mixture of Beliefs module. |
| `--num_components` | Number of evidential mixture components. Default in the paper: `3`. |
| `--use_fi_regularization` | Enables Fisher Information regularization. |
| `--fi_lambda` | Strength of FI regularization and FI-based modulation. |
| `--use_vos` | Enables Virtual Outlier Synthesis. |

## Model Variants

### GEM-CORE

GEM-CORE uses one evidential head and a learned energy-to-gate module. The energy signal is mapped to a bounded gate that modulates predictive probabilities and helps suppress overconfident predictions in low-support regions.

### GEM-MIX

GEM-MIX extends GEM-CORE with multiple evidential heads and a learned router. The mixture structure captures multi-modal epistemic uncertainty while preserving single-pass inference.

### GEM-FI

GEM-FI adds Fisher Information-based regularization and modulation to GEM-MIX. The FI terms stabilize mixture allocation, discourage head collapse, and improve uncertainty calibration and OOD separation.

## Results

Experimental results are saved in the `saved_results` directory and organized by dataset and model configuration.

Typical outputs include:

- classification accuracy,
- confidence calibration metrics,
- Brier score,
- expected calibration error,
- OOD detection AUPR,
- OOD detection AUROC,
- aleatoric uncertainty scores,
- epistemic uncertainty scores,
- mixture-aware mutual information.

## Reproducing Paper Experiments

The main experiments in the paper use:

| Dataset | Backbone | Epochs | Batch size | Learning rate | Components | FI lambda |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | ConvNet3C3F | 50 | 64 | `5e-4` | 3 | 0.3 |
| CIFAR-10 | ResNet18 | 100 | 128 | `1e-3` | 3 | 0.1 |

For CIFAR-10 GEM-FI:

```bash
python main.py --ID_dataset CIFAR-10 --backbone ResNet18 --batch_size 128 \
  --num_epochs 100 --learning_rate 1e-3 --dropout_rate 0.1 --reg_param 1e-4 \
  --use_mob --num_components 3 --use_fi_regularization --fi_lambda 0.1
```

For MNIST GEM-FI:

```bash
python main.py --ID_dataset MNIST --backbone ConvNet3C3F --batch_size 64 \
  --num_epochs 50 --learning_rate 5e-4 --dropout_rate 0.05 --reg_param 1e-3 \
  --use_mob --num_components 3 --use_fi_regularization --fi_lambda 0.3
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{mohammed2026gemfi,
  title     = {GEM-FI: Gated Evidential Mixtures with Fisher Modulation},
  author    = {Mohammed, Marco Mustafa and Daneshfar, Fatemeh and Li{\`o}, Pietro},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

## License

Add the license for this repository here.

Recommended options for academic code releases include:

- MIT License
- Apache License 2.0
- BSD 3-Clause License

## Contact

For questions about the paper or code, please contact:

**Marco Mustafa Mohammed**  
Email: `marco.mohammed@epu.edu.iq`
