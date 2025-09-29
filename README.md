# Speech Feature Extraction for ASR

A comprehensive implementation of acoustic feature extraction algorithms used in Automatic Speech Recognition (ASR) systems. This project computes **Log Mel Filterbank (LMF)** and **Mel-Frequency Cepstral Coefficients (MFCC)** features from raw audio signals.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Pipeline](#algorithm-pipeline)
- [Mathematical Background](#mathematical-background)
- [Implementation Details](#implementation-details)
- [Parameters](#parameters)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [References](#references)

---

## Overview

This project implements the fundamental signal processing pipeline for extracting acoustic features from speech audio. These features are critical for training and deploying speech recognition systems, as they transform raw audio waveforms into compact, discriminative representations that capture phonetic content while discarding irrelevant information.

### Why Feature Extraction?

Raw audio waveforms are high-dimensional and contain redundant information. Feature extraction:
- **Reduces dimensionality**: From 16,000 samples/second to ~100 feature vectors/second
- **Mimics human perception**: Uses perceptually-motivated Mel frequency scale
- **Improves robustness**: Removes speaker-specific characteristics
- **Enhances separability**: Creates representations optimized for phonetic discrimination

---

## Features

✅ **Complete Feature Extraction Pipeline**
- Pre-emphasis filtering
- Overlapping frame segmentation
- Power spectrum computation via FFT
- Mel-scale filterbank application
- Logarithmic compression
- Discrete Cosine Transform (DCT)
- Cepstral liftering

✅ **Two Feature Types**
- **LMF (Log Mel Filterbank)**: 80-dimensional features per frame
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 23-dimensional features per frame

✅ **Visualization Tools**
- Power spectrogram
- LMF feature map
- MFCC feature map
- Side-by-side comparison plots

✅ **Validated Implementation**
- Tested against ground truth features
- Numerical precision verification

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

Or install all at once:

```bash
pip install numpy==1.21.0 scipy==1.7.0 matplotlib==3.4.2
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

### Basic Usage

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

### Advanced Usage: Custom Parameters

```python
# High-resolution MFCC (40 filters, 13 coefficients)
mfcc_13 = compute_mfcc_feats(
    raw_signal=audio,
    window_length=0.032,      # Longer window for better frequency resolution
    overlap_length=0.012,
    sampling_rate=16000,
    num_mel_filters=40,
    num_ceps=13,
    ceplifter=22
)
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

## Algorithm Pipeline

The feature extraction process follows this sequence:

```
┌─────────────┐
│ Raw Audio   │ 16000 Hz, mono
│ Waveform    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Pre-emphasis│ High-pass filter: p[n] = x[n] - 0.95*x[n-1]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Framing     │ 25ms windows, 10ms overlap → ~60% overlap
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Windowing + │ Hamming window applied implicitly via FFT
│ FFT         │ NFFT = 512 (smallest power-of-2 ≥ 400 samples)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Power       │ |FFT|² / NFFT
│ Spectrum    │ Shape: (num_frames, 257)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Mel Filter  │ 80 triangular filters on Mel scale
│ Banks       │ Frequency warping: hz → mel
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Log         │ log(mel_features)
│ Compression │ LMF FEATURES (num_frames, 80)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ DCT-II      │ Decorrelates filter outputs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Keep First  │ Retain first 23 coefficients
│ 23 Coeffs   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Liftering   │ Sinusoidal weighting function
│             │ MFCC FEATURES (num_frames, 23)
└─────────────┘
```

---

## Mathematical Background

### 1. Mel Scale Conversion

The **Mel scale** is a perceptual scale of pitches judged by listeners to be equal in distance:

```
mel = 2595 × log₁₀(1 + hz/700)
hz = 700 × (10^(mel/2595) - 1)
```

**Why?** The human auditory system has better frequency resolution at lower frequencies. The Mel scale linearizes this perception.

### 2. Pre-emphasis

Applies a first-order high-pass filter:

```
p[n] = x[n] - α × x[n-1]    where α = 0.95
```

**Purpose:**
- Boosts high frequencies (typically attenuated during speech production)
- Balances frequency spectrum
- Improves SNR for high-frequency phonemes (e.g., fricatives)

### 3. Mel Filterbank

Triangular filters spaced uniformly on the Mel scale:

```
           /\
          /  \
         /    \
        /      \
       /        \
    --/----------\--
    f[m-1]  f[m]  f[m+1]
```

**Filter response:**
```
H_m(k) = { (k - f[m-1])/(f[m] - f[m-1])     if f[m-1] ≤ k < f[m]
         { (f[m+1] - k)/(f[m+1] - f[m])     if f[m] ≤ k ≤ f[m+1]
         { 0                                 otherwise
```

### 4. Discrete Cosine Transform (DCT)

Applies DCT-II to decorrelate log-mel features:

```
c[n] = Σ(m=0 to M-1) X[m] × cos(πn(m + 0.5)/M)
```

**Why?**
- Adjacent mel filters overlap → correlated outputs
- DCT compacts energy into first few coefficients
- Reduces dimensionality (80 → 23 coefficients)

### 5. Cepstral Liftering

Applies a sinusoidal weighting to MFCC coefficients:

```
w[n] = 1 + (L/2) × sin(πn/L)    where L = 22
c'[n] = w[n] × c[n]
```

**Purpose:**
- De-emphasizes high-order MFCCs (often noisy)
- Emphasizes mid-range coefficients (carry phonetic information)

---

## Implementation Details

### Key Functions

#### `frame_with_overlap(signal, window_length, overlap_length)`
Segments audio into overlapping frames.

**Implementation highlights:**
- Uses sliding window with hop length = window_length - overlap_length
- Pads final incomplete frame with zeros
- Returns shape: `(num_frames, window_length)`

**Example:**
```python
# For 1-second audio at 16kHz:
signal = np.random.randn(16000)
frames = frame_with_overlap(signal, window_length=400, overlap_length=160)
# frames.shape = (67, 400)  → 67 frames of 400 samples each
```

#### `compute_powerspec(framed_signal, window_sample_length)`
Computes power spectrum via FFT.

**Implementation highlights:**
1. NFFT = smallest power-of-2 ≥ window_length (for computational efficiency)
2. Applies `np.fft.rfft()` (real FFT) to each frame
3. Computes power: `|FFT|² / NFFT`

**Example:**
```python
power_spec, nfft = compute_powerspec(frames, window_sample_length=400)
# power_spec.shape = (67, 257) where 257 = 512/2 + 1
# nfft = 512
```

#### `get_mel_bank(num_filters, lowfreq, highfreq, nfft, sampling_rate)`
Creates Mel-scale triangular filterbank.

**Process:**
1. Generate `num_filters + 2` evenly-spaced points on Mel scale
2. Convert Mel points back to Hz
3. Map Hz to FFT bin indices
4. Construct triangular filters with 50% overlap

**Example:**
```python
filterbank = get_mel_bank(num_filters=80, lowfreq=0, highfreq=8000,
                          nfft=512, sampling_rate=16000)
# filterbank.shape = (80, 257)
```

#### `get_mel_fbank_feat(power_spec, mel_filterbanks)`
Applies filterbank to power spectrum.

**Implementation:**
```python
mel_fbank = power_spec @ mel_filterbanks.T  # Matrix multiplication
mel_fbank[mel_fbank == 0] = eps             # Prevent log(0)
```

**Example:**
```python
mel_feats = get_mel_fbank_feat(power_spec, filterbank)
# mel_feats.shape = (67, 80)
```

#### `compute_lmf_feats()` and `compute_mfcc_feats()`
High-level wrapper functions that orchestrate the entire pipeline.

---

## Parameters

### Recommended Parameter Sets

#### Standard Speech Recognition (16kHz)
```python
window_length = 0.025       # 25ms
overlap_length = 0.01       # 10ms (15ms hop)
sampling_rate = 16000       # 16kHz
num_mel_filters = 80        # 80 filters
num_ceps = 23               # 23 MFCC coefficients
```

#### Telephony/Narrowband (8kHz)
```python
window_length = 0.025
overlap_length = 0.01
sampling_rate = 8000
mel_high_freq = 4000        # Nyquist frequency
num_mel_filters = 40
num_ceps = 13
```

#### High-Resolution (44.1kHz music)
```python
window_length = 0.046       # Longer window for better freq resolution
overlap_length = 0.023
sampling_rate = 44100
mel_high_freq = 22050
num_mel_filters = 128
num_ceps = 40
```

### Parameter Descriptions

| Parameter | Type | Description | Typical Range |
|-----------|------|-------------|---------------|
| `window_length` | float | Frame duration (seconds) | 0.020 - 0.040 |
| `overlap_length` | float | Overlap between frames (seconds) | 0.010 - 0.015 |
| `sampling_rate` | int | Audio sampling rate (Hz) | 8000, 16000, 44100 |
| `preemph` | bool | Apply pre-emphasis filter | True |
| `mel_low_freq` | int | Lowest frequency in filterbank (Hz) | 0, 64, 133 |
| `mel_high_freq` | int | Highest frequency in filterbank (Hz) | fs/2 |
| `num_mel_filters` | int | Number of triangular filters | 23, 40, 80, 128 |
| `num_ceps` | int | Number of MFCC coefficients to keep | 13, 23, 40 |
| `ceplifter` | int | Liftering parameter (L) | 22 |

### Time-Frequency Trade-offs

**Window Length:**
- **Longer (40ms)**: Better frequency resolution, worse time resolution
- **Shorter (20ms)**: Better time resolution, worse frequency resolution
- **Optimal**: 25ms captures 2-3 pitch periods for typical speech

**Overlap:**
- More overlap (75%) → Smoother features, more computation
- Less overlap (50%) → Faster computation, less redundancy
- Standard: 60% overlap (25ms window, 10ms hop)

---

## Visualization

The visualization function generates a three-panel comparison plot:

### Panel 1: Power Spectrogram
- **X-axis**: Time (seconds)
- **Y-axis**: Frequency (Hz) - **Linear scale**
- **Color**: Power (dB)
- **Shows**: Full frequency resolution with detailed harmonic structure

### Panel 2: Log Mel Filterbank (LMF)
- **X-axis**: Time (seconds)
- **Y-axis**: Mel filter index (0-79)
- **Color**: Log magnitude
- **Shows**: Mel-warped frequency axis with emphasis on lower frequencies

### Panel 3: MFCC Features
- **X-axis**: Time (seconds)
- **Y-axis**: Cepstral coefficient index (0-22)
- **Color**: Coefficient value
- **Shows**: Decorrelated, compact representation

### Interpreting the Plots

**Power Spectrogram:**
- Horizontal bands = speech formants (resonances of vocal tract)
- Vertical striations = pitch harmonics
- Dark regions = high energy (vowels)
- Light regions = low energy (consonants)

**LMF Features:**
- Smoother than spectrogram due to filterbank averaging
- Lower indices = low frequencies (more detail)
- Higher indices = high frequencies (less detail)
- Captures gross spectral shape

**MFCC Features:**
- First few coefficients (0-5) = overall spectral shape
- Mid-range coefficients (6-15) = phonetic details
- High coefficients (16-22) = fine spectral structure
- Most energy concentrated in first 10-12 coefficients

---

## Project Structure

```
coding1/
├── README.md                    # This file
├── handout.pdf                  # Assignment handout
│
└── feat_extract/
    ├── feat_extract.py          # Main implementation (386 lines)
    │   ├── Utility Functions
    │   │   ├── hz2mel()                    # Hz → Mel conversion
    │   │   ├── mel2hz()                    # Mel → Hz conversion
    │   │   ├── preemphasis()               # High-pass filter
    │   │   ├── get_log_feats()             # Logarithm wrapper
    │   │   └── lifter()                    # Cepstral liftering
    │   │
    │   ├── Core Functions
    │   │   ├── frame_with_overlap()        # Frame segmentation ⭐
    │   │   ├── compute_powerspec()         # FFT + power ⭐
    │   │   ├── get_mel_bank()              # Mel filterbank creation
    │   │   └── get_mel_fbank_feat()        # Apply filterbank ⭐
    │   │
    │   ├── High-Level Functions
    │   │   ├── compute_lmf_feats()         # Full LMF pipeline
    │   │   └── compute_mfcc_feats()        # Full MFCC pipeline
    │   │
    │   ├── Visualization
    │   │   └── visualize_features()        # 3-panel plot generator
    │   │
    │   └── Main Execution
    │       └── if __name__ == "__main__"   # Test & validation
    │
    ├── analysis.txt             # Feature comparison observations
    ├── feature_visualization.png # Generated visualization (300 DPI)
    │
    └── example_data/
        ├── example_audio.wav    # Test audio (270 KB, 16kHz mono)
        └── example_feats.npz    # Ground truth features (465 KB)
                                 # Contains: 'lmel' and 'mfcc' arrays
```

**⭐ = Functions implemented by Anshul Kumar**

---

## Testing

### Validation Method

The code validates correctness by comparing computed features against pre-computed ground truth:

```python
# Load ground truth
feat_arrays = np.load("example_data/example_feats.npz")
lmf_truth = feat_arrays["lmel"]
mfcc_truth = feat_arrays["mfcc"]

# Compute features
my_lmf = compute_lmf_feats(audio, ...)
my_mfcc = compute_mfcc_feats(audio, ...)

# Validate with numerical tolerance
assert np.allclose(my_lmf, lmf_truth, rtol=1e-5, atol=1e-8)
assert np.allclose(my_mfcc, mfcc_truth, rtol=1e-5, atol=1e-8)
```

### Running Tests

```bash
cd feat_extract
python feat_extract.py
```

**Expected output:**
```
---------- Success! ----------
Visualization saved as feature_visualization.png
```

### Manual Testing

```python
# Test with your own audio
sampling_rate, audio = wavfile.read("path/to/your/audio.wav")

# Audio must be:
# - Mono (single channel)
# - 16kHz sampling rate (or adjust parameters)
# - 16-bit PCM format

# Compute features
lmf = compute_lmf_feats(audio, window_length=0.025, overlap_length=0.01,
                         sampling_rate=sampling_rate)
print(f"Feature shape: {lmf.shape}")
# Expected: (num_frames, 80) where num_frames ≈ audio_duration / 0.01
```

### Debugging Tips

**Issue: Assertion fails for LMF**
- Check frame segmentation (ensure last frame is padded)
- Verify NFFT calculation (must be power-of-2)
- Confirm filterbank construction (check bin indices)

**Issue: Assertion fails for MFCC**
- Ensure DCT uses `type=2, norm='ortho'`
- Verify coefficient selection (first `num_ceps` only)
- Check liftering function (sinusoidal window)

**Issue: "Cannot take log of zero"**
- Ensure `get_mel_fbank_feat()` replaces zeros with epsilon
- Check filterbank coverage (some bins may be unfiltered)

---

## References

### Papers & Books

1. **Davis, S., & Mermelstein, P. (1980)**
   *"Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences"*
   IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366.
   [The original MFCC paper]

2. **Rabiner, L. R., & Schafer, R. W. (2007)**
   *"Introduction to Digital Speech Processing"*
   Foundations and Trends in Signal Processing, 1(1-2), 1-194.
   [Comprehensive DSP for speech]

3. **Gold, B., Morgan, N., & Ellis, D. (2011)**
   *"Speech and Audio Signal Processing"*
   Wiley. [Textbook covering all aspects]

### Online Resources

- [Mel Frequency Cepstral Coefficients (MFCCs) Tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- [Scipy FFT Documentation](https://docs.scipy.org/doc/scipy/reference/fft.html)
- [NumPy Audio Processing Guide](https://numpy.org/doc/stable/reference/routines.fft.html)

### Course Materials

- **18-781 Speech Recognition and Understanding** (Fall 2025)
  Carnegie Mellon University
  Instructor: Prof. Shinji Watanabe
  Course website: https://www.wavlab.org/

---

## Common Issues & FAQ

### Q1: Why are my features different from the ground truth?

**A:** Ensure you're using:
- Correct NFFT calculation (power-of-2)
- Proper frame padding (zero-pad last frame)
- Epsilon replacement before taking log
- Correct DCT parameters (`type=2, norm='ortho'`)

### Q2: What's the difference between LMF and MFCC?

**A:**
- **LMF**: Direct log of mel filterbank outputs (80-dim)
- **MFCC**: DCT of LMF (23-dim), decorrelated and more compact
- **When to use**: LMF for deep learning (neural nets learn DCT), MFCC for GMM-HMM systems

### Q3: Can I use this for music/environmental sounds?

**A:** Yes, but:
- Music may need higher frequency range (up to 22kHz)
- More filters (128-256) capture richer timbre
- Longer windows (46ms) improve frequency resolution

### Q4: How do I handle different sampling rates?

**A:** Adjust parameters proportionally:
```python
# For 8kHz audio:
mel_high_freq = 4000  # Nyquist frequency
num_mel_filters = 40  # Fewer filters for narrower bandwidth
```

### Q5: Why 23 MFCC coefficients?

**A:** Historical convention:
- 0th coefficient = log energy (often discarded)
- Coefficients 1-12 = phonetic information
- Coefficients 13-22 = speaker characteristics
- Beyond 23: diminishing returns for ASR

### Q6: What's the computational complexity?

**A:**
- **Framing**: O(N) where N = signal length
- **FFT**: O(T × NFFT × log(NFFT)) where T = num_frames
- **Filterbank**: O(T × NFFT × M) where M = num_filters
- **DCT**: O(T × M × C) where C = num_ceps
- **Total**: O(T × NFFT × log(NFFT)) [FFT dominates]

---

## Performance Optimization Tips

1. **Use librosa** for production:
   ```python
   import librosa
   mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=23)
   ```

2. **Vectorize operations**: Already done in this implementation using NumPy

3. **Batch processing**: Process multiple audio files in parallel
   ```python
   from multiprocessing import Pool
   with Pool(8) as p:
       features = p.map(compute_mfcc_feats, audio_list)
   ```

4. **GPU acceleration**: Use PyTorch/TensorFlow for large-scale processing

---

## License

This project is part of the coursework for **18-781 Speech Recognition and Understanding** at Carnegie Mellon University.

**Academic Integrity Notice**: If you are a current student in this course, please adhere to the collaboration policies outlined in the syllabus.

---

## Acknowledgments

- **Course**: 18-781 Speech Recognition and Understanding, Carnegie Mellon University
- **Instructor**: Prof. Shinji Watanabe
- **Course Staff**: For providing the assignment framework and ground truth validation data
- **Libraries**: NumPy, SciPy, Matplotlib for robust scientific computing tools
- **References**: Davis & Mermelstein (1980) for the foundational MFCC algorithm

---

**Last Updated**: September 2025
**Course**: 18-781 Speech Recognition and Understanding (Prof. Shinji Watanabe)
**Institution**: Carnegie Mellon University