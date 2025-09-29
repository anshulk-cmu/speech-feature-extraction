# Contributing to Speech Feature Extraction for ASR

Thank you for your interest in contributing! This project was originally developed as coursework for CMU's 18-781 Speech Recognition and Understanding course and is now maintained as an educational resource.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](../../issues) section
2. If not, create a new issue with:
   - Clear description of the problem/suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (Python version, OS, package versions)

### Submitting Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/anshulk-cmu/speech-feature-extraction.git
   cd speech-feature-extraction
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, readable code
   - Follow existing code style and conventions
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   ```bash
   cd feat_extract
   python feat_extract.py
   # Ensure all tests pass
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use descriptive variable and function names
- Add docstrings for all functions
- Keep functions focused and modular

### Documentation

- Update README.md if adding new features
- Include docstrings with parameter descriptions
- Add inline comments for complex algorithms
- Update examples if changing APIs

### Testing

- Ensure existing tests pass
- Add tests for new functionality
- Validate against ground truth where possible
- Test with different audio formats and parameters

## 💡 Ideas for Contributions

### High Priority

- [ ] Add delta and delta-delta (velocity/acceleration) features
- [ ] Implement command-line interface
- [ ] Add more comprehensive unit tests
- [ ] Support for real-time streaming audio
- [ ] Integration examples with popular ML frameworks

### Medium Priority

- [ ] Pitch tracking (F0) extraction
- [ ] Energy normalization
- [ ] Voice Activity Detection (VAD)
- [ ] Support for stereo/multi-channel audio
- [ ] Performance benchmarking suite

### Low Priority

- [ ] Additional feature types (PLP, RASTA-PLP)
- [ ] GUI for interactive visualization
- [ ] Docker containerization
- [ ] Jupyter notebook tutorials
- [ ] Comparison with librosa/python_speech_features

## ⚠️ Important Notes

### Academic Integrity

This project is educational in nature. If you're a student:
- **DO NOT** submit this code as your own coursework
- Use it to **understand concepts**, not to copy solutions
- Follow your institution's academic integrity policies

### Attribution

- Original implementation: Anshul Kumar (CMU 18-781, Fall 2025)
- Based on: Davis & Mermelstein (1980) MFCC algorithm
- Course framework provided by CMU course staff

## 📞 Questions?

- Open an issue for bugs or feature requests
- Email: anshulk@andrew.cmu.edu for other inquiries
- Check existing documentation in README.md

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make this project better! 🙏