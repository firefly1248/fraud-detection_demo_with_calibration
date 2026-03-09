"""Setup script for calibrated-binary-classifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="calibrated-binary-classifier",
    version="1.0.0",
    author="Ilia Ekhlakov",
    description="Production-ready binary classification framework with advanced calibration (Venn-ABERS, isotonic) and temporal validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/calibrated-binary-classifier",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "black>=23.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
        ]
    },
    keywords=[
        "machine-learning",
        "fraud-detection",
        "calibration",
        "venn-abers",
        "conformal-prediction",
        "binary-classification",
        "lightgbm",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/calibrated-binary-classifier/issues",
        "Source": "https://github.com/yourusername/calibrated-binary-classifier",
        "Documentation": "https://github.com/yourusername/calibrated-binary-classifier/blob/main/CLAUDE.md",
    },
)
