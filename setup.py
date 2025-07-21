"""
Setup script for VoiceAccess
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="voiceaccess",
    version="0.1.0",
    author="OpenImpactAI",
    author_email="contact@openimpactai.org",
    description="Automatic Speech Recognition for Low-Resource and Endangered Languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openimpactai/VoiceAccess",
    project_urls={
        "Bug Tracker": "https://github.com/openimpactai/VoiceAccess/issues",
        "Documentation": "https://github.com/openimpactai/VoiceAccess/wiki",
        "Source Code": "https://github.com/openimpactai/VoiceAccess",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "gpu": [
            "nvidia-ml-py>=11.525.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voiceaccess=voiceaccess.cli:main",
            "voiceaccess-train=voiceaccess.cli.train:main",
            "voiceaccess-transcribe=voiceaccess.cli.transcribe:main",
            "voiceaccess-api=voiceaccess.api.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "voiceaccess": ["configs/*.yaml", "configs/*.json"],
    },
)