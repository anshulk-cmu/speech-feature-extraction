# Speech Feature Extraction for ASR

A comprehensive implementation of acoustic feature extraction algorithms used in Automatic Speech Recognition (ASR) systems. This project computes **Log Mel Filterbank (LMF)** and **Mel-Frequency Cepstral Coefficients (MFCC)** features from raw audio signals.

Developed as coursework for **18-781 Speech Recognition and Understanding** at Carnegie Mellon University (Fall 2025, Prof. Shinji Watanabe).

---

## Overview

This project implements the fundamental signal processing pipeline for extracting acoustic features from speech audio. These features transform raw audio waveforms into compact, discriminative representations that capture phonetic content while discarding irrelevant information.

### Why Feature Extraction?

Raw audio waveforms are high-dimensional and contain redundant information. Feature extraction:
- **Reduces dimensionality**: From 16,000 samples/second to ~100 feature vectors/second
- **Mimics human perception**: Uses perceptually-motivated Mel frequency scale
- **Improves robustness**: Removes speaker-specific characteristics
- **Enhances separability**: Creates representations optimized for phonetic discrimination

---

## Features

- ✅ Complete feature extraction pipeline (pre-emphasis, framing, FFT, mel-filtering, DCT)
- ✅ Log Mel Filterbank (LMF): 80-dimensional features per frame
- ✅ Mel-Frequency Cepstral Coefficients (MFCC): 23-dimensional features per frame
- ✅ Visualization tools comparing spectrograms, LMF, and MFCC
- ✅ Validated against ground truth features

---

## Installation

### Prerequisites

```bash
Python >= 3.7
```

### Dependencies

```bash
pip install numpy scipy matplotlib
```

### Verify Installation

```bash
cd feat_extract
python feat_extract.py
```

Expected output:
```
---------- Success! ----------
Visualization saved as feature_visualization.png
```

---

## Usage

### Basic Example

```python
from scipy.io import wavfile
from feat_extract import compute_lmf_feats, compute_mfcc_feats

# Load audio
sampling_rate, audio = wavfile.read("example_data/example_audio.wav")

# Compute LMF features (80 filters)
lmf_features = compute_lmf_feats(
    raw_signal=audio,
    window_length=0.025,      # 25ms windows
    overlap_length=0.01,      # 10ms overlap
    sampling_rate=sampling_rate,
    preemph=True,
    mel_low_freq=0,
    mel_high_freq=8000,
    num_mel_filters=80
)

# Compute MFCC features (23 coefficients)
mfcc_features = compute_mfcc_feats(
    raw_signal=audio,
    window_length=0.025,
    overlap_length=0.01,
    sampling_rate=sampling_rate,
    preemph=True,
    mel_low_freq=0,
    mel_high_freq=8000,
    num_mel_filters=80,
    num_ceps=23,
    ceplifter=22
)

print(f"LMF shape: {lmf_features.shape}")   # (num_frames, 80)
print(f"MFCC shape: {mfcc_features.shape}") # (num_frames, 23)
```

### Generating Visualizations

```python
from feat_extract import visualize_features

visualize_features(
    audio=audio,
    lmf_features=lmf_features,
    mfcc_features=mfcc_features,
    sampling_rate=16000,
    window_length=0.025,
    overlap_length=0.01,
    output_filename="my_features.png"
)
```

---

## How It Works

The feature extraction pipeline:

```
Raw Audio (16kHz mono)
    ↓
Pre-emphasis Filter (boost high frequencies)
    ↓
Frame Segmentation (25ms windows, 10ms overlap)
    ↓
FFT → Power Spectrum
    ↓
Mel Filterbank (80 triangular filters)
    ↓
Logarithmic Compression → LMF Features (80-dim)
    ↓
Discrete Cosine Transform (DCT)
    ↓
Keep First 23 Coefficients + Liftering → MFCC Features (23-dim)
```

### Key Algorithms

**Mel Scale Conversion**
```
mel = 2595 × log₁₀(1 + hz/700)
hz = 700 × (10^(mel/2595) - 1)
```
The Mel scale linearizes human frequency perception (better resolution at low frequencies).

**Pre-emphasis Filter**
```
p[n] = x[n] - 0.95 × x[n-1]
```
Boosts high frequencies typically attenuated during speech production.

**Mel Filterbank**
Triangular filters spaced uniformly on the Mel scale, with 50% overlap between adjacent filters.

**Discrete Cosine Transform (DCT)**
Decorrelates log-mel features and compacts energy into first few coefficients.

**Cepstral Liftering**
```
w[n] = 1 + (L/2) × sin(πn/L)    where L = 22
```
Applies sinusoidal weighting to emphasize mid-range MFCC coefficients.

---

## Implementation

### Core Functions

**`frame_with_overlap(signal, window_length, overlap_length)`**
Segments audio into overlapping frames with zero-padding for incomplete final frame.

**`compute_powerspec(framed_signal, window_sample_length)`**
Computes power spectrum via FFT with NFFT = smallest power-of-2 ≥ window_length.

**`get_mel_bank(num_filters, lowfreq, highfreq, nfft, sampling_rate)`**
Creates Mel-scale triangular filterbank.

**`get_mel_fbank_feat(power_spec, mel_filterbanks)`**
Applies filterbank to power spectrum via matrix multiplication.

**`compute_lmf_feats()` and `compute_mfcc_feats()`**
High-level wrappers orchestrating the entire pipeline.

---

## Parameters

### Standard Configuration (16kHz Speech)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_length` | 0.025 | Frame duration (25ms) |
| `overlap_length` | 0.01 | Overlap between frames (10ms) |
| `sampling_rate` | 16000 | Audio sampling rate (Hz) |
| `preemph` | True | Apply pre-emphasis filter |
| `mel_low_freq` | 0 | Lowest frequency in filterbank |
| `mel_high_freq` | 8000 | Highest frequency (sampling_rate/2) |
| `num_mel_filters` | 80 | Number of triangular filters |
| `num_ceps` | 23 | Number of MFCC coefficients |
| `ceplifter` | 22 | Liftering parameter |

---

## Visualization

The `visualize_features()` function generates a three-panel comparison:

1. **Power Spectrogram**: Full frequency resolution with detailed harmonic structure
2. **Log Mel Filterbank (LMF)**: Mel-warped frequency axis emphasizing lower frequencies
3. **MFCC Features**: Decorrelated, compact representation with energy concentrated in first coefficients

---

## Project Structure

```
coding1/
├── README.md
├── LICENSE
├── .gitignore
│
└── feat_extract/
    ├── feat_extract.py          # Main implementation (386 lines)
    ├── analysis.txt             # Feature comparison observations
    ├── check_setup.py           # Dependency verification script
    ├── feature_visualization.png # Generated visualization
    │
    └── example_data/
        ├── example_audio.wav    # Test audio (16kHz mono)
        └── example_feats.npz    # Ground truth features
```

---

## Testing

The code validates correctness against pre-computed ground truth:

```bash
cd feat_extract
python feat_extract.py
```

**Expected output:**
```
---------- Success! ----------
Visualization saved as feature_visualization.png
```

### Validation Method

```python
# Load ground truth
feat_arrays = np.load("example_data/example_feats.npz")
lmf_truth = feat_arrays["lmel"]
mfcc_truth = feat_arrays["mfcc"]

# Validate with numerical tolerance
assert np.allclose(my_lmf, lmf_truth, rtol=1e-5, atol=1e-8)
assert np.allclose(my_mfcc, mfcc_truth, rtol=1e-5, atol=1e-8)
```

---

## References

### Papers

1. **Davis, S., & Mermelstein, P. (1980)**  
   *"Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences"*  
   IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366.

2. **Rabiner, L. R., & Schafer, R. W. (2007)**  
   *"Introduction to Digital Speech Processing"*  
   Foundations and Trends in Signal Processing, 1(1-2), 1-194.

### Course

**18-781 Speech Recognition and Understanding** (Fall 2025)  
Carnegie Mellon University  
Instructor: Prof. Shinji Watanabe  
Course website: https://www.wavlab.org/

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Anshul Kumar**  
Email: anshulk@andrew.cmu.edu  
Carnegie Mellon University

---

**Academic Integrity Notice**: This code is shared for educational purposes. Current students should not copy it for coursework.