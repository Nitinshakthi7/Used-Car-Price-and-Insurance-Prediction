"""
📦 Requirements Installer
Run this script FIRST before running main.py to install all required libraries.

Usage:
    python requirements.py
"""
import subprocess
import sys


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "joblib",
]


def install_packages():
    print("=" * 55)
    print("   INSTALLING REQUIRED PACKAGES")
    print("=" * 55)

    for package in REQUIRED_PACKAGES:
        print(f"\n[INSTALLING] {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"]
        )
        print(f"  ✅ {package} installed successfully")

    print("\n" + "=" * 55)
    print("   ALL PACKAGES INSTALLED SUCCESSFULLY!")
    print("   You can now run:  python main.py")
    print("=" * 55)


if __name__ == "__main__":
    install_packages()
