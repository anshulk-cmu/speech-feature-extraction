#!/usr/bin/env python3
"""
Setup verification script for Speech Feature Extraction
Checks if all dependencies are properly installed and working.
"""

import sys

def check_python_version():
    """Check if Python version is 3.7 or higher."""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.7+)")
        return False

def check_numpy():
    """Check if NumPy is installed and working."""
    print("Checking NumPy...", end=" ")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        return True
    except ImportError:
        print("✗ Not installed")
        return False

def check_scipy():
    """Check if SciPy is installed and working."""
    print("Checking SciPy...", end=" ")
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
        return True
    except ImportError:
        print("✗ Not installed")
        return False

def check_matplotlib():
    """Check if Matplotlib is installed and working."""
    print("Checking Matplotlib...", end=" ")
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        return True
    except ImportError:
        print("✗ Not installed (optional for visualization)")
        return True  # Optional dependency

def check_files():
    """Check if required files exist."""
    print("Checking required files...", end=" ")
    import os
    required_files = [
        'feat_extract.py',
        'example_data/example_audio.wav',
        'example_data/example_feats.npz'
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)

    if not missing:
        print("✓ All files present")
        return True
    else:
        print(f"✗ Missing: {', '.join(missing)}")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("Speech Feature Extraction - Setup Verification")
    print("=" * 60)
    print()

    checks = [
        check_python_version(),
        check_numpy(),
        check_scipy(),
        check_matplotlib(),
        check_files()
    ]

    print()
    print("=" * 60)

    if all(checks):
        print("✓ All checks passed! You're ready to go.")
        print("\nRun the following command to test the implementation:")
        print("  python feat_extract.py")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("\n  pip install numpy scipy matplotlib")

    print("=" * 60)

    return all(checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)