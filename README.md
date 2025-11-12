# UDSClassification

This repository contains the implementation of deep learning models for Up-Down States (UDS) classification. The code provides tools for training and evaluating models that classify UDS from LFP data (ABF files).

## Overview

This implementation provides deep learning models for classifying UDS from LFPs. The main features include:

- Deep learning-based UDS classification
- Support for multiple model architectures (Transformer, CNN-RNN)
- Data preprocessing (downsampling, filtering, STFT transformation)
- Model training and evaluation
- Cross-validation for model assessment

## Installation

### Requirements

- Python 3.7 or higher
- CUDA-capable GPU (recommended; CPU execution is also supported)

### Dependencies

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

### PyTorch Installation

PyTorch should be installed according to the [official PyTorch website](https://pytorch.org/) based on your system configuration.

For CUDA-enabled installations:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Preparation

### Data Format

- **Data files**: ABF format (Axon Binary Format)
- **Label files**: NumPy array format (.npy)

### Directory Structure

Data files should be organized as follows:

```
path/to/data/
  ├── file1_data.abf
  ├── file1_data_label.npy
  ├── file2_data.abf
  ├── file2_data_label.npy
  └── ...
```

### Configuration

In each script (`train_model.py`, `train_models.py`, `eval_models.py`), the path configuration section should be modified to specify the actual file paths:

```python
# Data file directory
DATA_DIR = "path/to/data"

# List of data file names (.abf file paths)
filenames = [
    os.path.join(DATA_DIR, "file1_data.abf"),
    os.path.join(DATA_DIR, "file2_data.abf"),
    # Add actual file paths here
]

# Label file suffix pattern
LABEL_SUFFIX = "_label.npy"
```

### Data Selection

Time ranges for each data file are specified using `used_idx`:

```python
# Time ranges (in seconds) to use from each data file
# Format: [[start1, end1], [start2, end2], ...]
# Files with [start, end] = [0, 0] are skipped
used_idx = [
    [0, 940],   # Time range for file1_data.abf (in seconds)
    [0, 1250],  # Time range for file2_data.abf (in seconds)
    # Add actual time ranges here
]
```

## Usage

### 1. Single Model Training

A single model can be trained using `train_model.py`:

```bash
python train_model.py
```

**Configuration parameters**:
- Data file paths
- Spectrogram file paths (if using pre-computed STFT data)
- Output directory
- Model hyperparameters (number of layers, input size, learning rate, etc.)
- Training/validation data indices

### 2. Cross-Validation Training

Multiple models are trained using cross-validation with `train_models.py`:

```bash
python train_models.py
```

**Features**:
- Performs 3-fold cross-validation
- Trains separate models for each fold
- Computes STFT from raw data (does not use pre-computed STFT files)

### 3. Model Evaluation

Trained models are evaluated using `eval_models.py`:

```bash
python eval_models.py
```

**Evaluation outputs**:
- Uses multiple models trained via cross-validation
- Generates predictions on test data
- Computes Up/Down state coincidence rates
- Saves results (`coins.pkl`, `up_states.pkl`)

### Path Configuration

The following paths should be configured at the beginning of each script according to your environment:

```python
# Output directory
OUTPUT_DIR = "path/to/output"

# Checkpoint subdirectory
CHECKPOINT_SUBDIR = "transformers"

# Spectrogram file path (if used)
SPECTROGRAM_DIR = "path/to/spectrogram"
stft_path = os.path.join(SPECTROGRAM_DIR, "stft.pkl")
```

## Model Architectures

### CNN + RNN Model (CnnRnnModel)

The CnnRnnModel combines convolutional neural network (CNN) and recurrent neural network (RNN) components. The model first extracts features from spectrograms using 2D convolutions, then processes temporal sequences using LSTM or GRU units.

**Architecture**:
- 2D convolutional layer with batch normalization and ReLU activation
- LSTM or GRU layer for temporal processing
- Linear output layer for binary classification

**Parameters**:
- `freq_bins`: Number of frequency bins (default: 256)
- `time_steps`: Number of time steps (default: 2000)
- `conv_filters`: Number of convolutional filters (default: 32)
- `rnn_units`: Number of RNN hidden units (default: 64)
- `rnn_type`: Type of RNN ("lstm" or "gru", default: "lstm")

**Implementation details**:
The convolutional layer processes the spectrogram with a 5×5 kernel and same padding. The output is reshaped into temporal sequences and fed into the RNN layer. The final hidden states are used for binary classification at each time step.

### Transformer Model (TransformerEncoderModel)

The TransformerEncoderModel is based on the Transformer encoder architecture. The model processes time-frequency representations of neural signals and learns temporal dependencies through multi-head self-attention mechanisms.

**Architecture**:
- Multi-head self-attention with configurable number of heads
- Optional positional encoding
- Stacked Transformer encoder layers
- Linear output layer for binary classification

**Parameters**:
- `input_size`: Input feature dimension (default: 256)
- `n_head`: Number of attention heads (default: 4)
- `dim_ff`: Feedforward layer dimension (default: 512)
- `num_layers`: Number of Transformer encoder layers (default: 3)
- `pe`: Whether to use positional encoding (default: False)
- `max_len`: Maximum sequence length for positional encoding (default: 2000)

**Implementation details**:
The input spectrogram is reshaped into sequences of feature vectors. The Transformer encoder processes these sequences, and the output is passed through a linear layer to produce binary predictions for each time step.

## Data Preprocessing

### Data Loading

Local field potential (LFP) data are loaded from ABF files using the `pyabf` library. Label files containing Up-Down state annotations are loaded as NumPy arrays (.npy format). The data loading process extracts channel 1 from the ABF files, which corresponds to the LFP signal.

### Time Range Selection

Time segments are extracted from each data file according to the `used_idx` specification. Each entry in `used_idx` specifies the start and end times (in seconds) for the corresponding data file. Files with `[start, end] = [0, 0]` are excluded from further processing.

### Downsampling

Data are downsampled from the original sampling rate (typically 20,000 Hz) to a target rate (typically 500 Hz) using averaging. The downsampling factor is calculated as the ratio of the original and target sampling rates. Data are reshaped into non-overlapping windows, and the mean value within each window is computed.

### Bandpass Filtering

A second-order Butterworth bandpass filter is applied to the downsampled data. The default cutoff frequencies are 0.1 Hz (low cutoff) and 200 Hz (high cutoff). Filtering is performed using `filtfilt` to achieve zero-phase filtering, which eliminates phase distortion.

### Z-Score Normalization

Each data segment is normalized using z-score normalization, where the mean is subtracted and the result is divided by the standard deviation. This normalization step standardizes the data distribution and improves training stability.

### STFT Transformation

Short-Time Fourier Transform (STFT) is computed to obtain time-frequency representations of the signals. The STFT parameters are configured to yield a specified number of frequency bins (default: 128). The real and imaginary parts of the STFT coefficients are treated as separate channels in the input representation.

### Spectrogram Computation

Alternatively, spectrograms can be computed using continuous wavelet transform (CWT) with the complex Morlet wavelet (cmor1.5-2). The wavelet transform provides time-frequency localization and is computed across logarithmically spaced frequency scales. Similar to STFT, the real and imaginary parts of the wavelet coefficients are used as separate input channels.

## Training Procedure

### Loss Function

Binary cross-entropy with logits is used as the loss function. The model outputs logits, which are converted to probabilities using the sigmoid function during loss computation.

### Optimization

Models are optimized using the Adam optimizer with a learning rate of 5×10⁻⁵ (default). Weight decay can be optionally applied. Training continues for a specified number of epochs or until early stopping is triggered.

### Early Stopping

Early stopping is implemented to prevent overfitting. Training is terminated if the validation loss does not improve for a specified number of epochs (patience parameter, default: 20-50 epochs depending on the script).

### Evaluation Metrics

Model performance is evaluated using coincidence rates, which measure the agreement between predicted and ground truth Up-Down states. Two coincidence rates are computed:
- Up-state coincidence: Fraction of time points where both predicted and ground truth indicate up states, relative to all time points where at least one indicates an up state
- Down-state coincidence: Fraction of time points where both predicted and ground truth indicate down states, relative to all time points where at least one indicates a down state

## Inference

During inference, overlapping windows are processed with a specified stride. Predictions from overlapping windows are aggregated by voting, where the final prediction for each time point is determined by majority voting across all windows covering that time point. A duration filter is applied to remove state transitions shorter than a minimum duration (default: 40 ms).
