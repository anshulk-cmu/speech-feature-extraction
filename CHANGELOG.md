# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-09-29

### Added
- Complete implementation of LMF (Log Mel Filterbank) feature extraction
- Complete implementation of MFCC (Mel-Frequency Cepstral Coefficients) extraction
- Core DSP functions:
  - `frame_with_overlap()` - Signal segmentation with overlapping windows
  - `compute_powerspec()` - FFT-based power spectrum computation
  - `get_mel_fbank_feat()` - Mel filterbank application
- Visualization function for comparing spectrograms, LMF, and MFCC
- Validation tests against ground truth features
- Comprehensive README with mathematical background and usage examples
- Setup verification script (`check_setup.py`)
- MIT License
- `.gitignore` for Python projects
- `CONTRIBUTING.md` with contribution guidelines
- `GITHUB_SETUP.md` with publishing instructions
- Academic integrity notice

### Implemented
- Pre-emphasis filtering with configurable coefficient
- Mel-scale frequency conversion (Hz ↔ Mel)
- Triangular mel filterbank generation
- Discrete Cosine Transform (DCT-II) for MFCC
- Cepstral liftering
- Zero-padding for incomplete frames
- Epsilon handling for log operations

### Documentation
- Complete algorithm pipeline flowchart
- Mathematical derivations for all transformations
- Parameter tuning guidelines
- Troubleshooting and FAQ section
- Performance optimization tips
- Example applications and use cases
- Future improvement roadmap

### Tested
- Validation against provided ground truth features
- Numerical precision within tolerance (rtol=1e-5, atol=1e-8)
- Example audio processing and visualization generation

### Features
- Support for 16kHz mono audio (configurable)
- 80-dimensional LMF features (configurable)
- 23-dimensional MFCC features (configurable)
- Customizable window length and overlap
- Publication-quality visualizations (300 DPI)

---

## Future Releases (Planned)

### [1.1.0] - Planned
- [ ] Delta and delta-delta features
- [ ] Command-line interface
- [ ] Batch processing support
- [ ] Real-time streaming audio

### [1.2.0] - Planned
- [ ] Pitch tracking (F0 extraction)
- [ ] Energy normalization
- [ ] Voice Activity Detection (VAD)
- [ ] Stereo/multi-channel support

### [2.0.0] - Planned
- [ ] Deep learning framework integration
- [ ] Additional feature types (PLP, RASTA-PLP)
- [ ] GUI for interactive visualization
- [ ] Docker containerization

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for ways to contribute.

## Author

**Anshul Kumar**
- Email: anshulk@andrew.cmu.edu
- Course: 18-781 Speech Recognition and Understanding, CMU
- Instructor: Prof. Shinji Watanabe
- Semester: Fall 2025