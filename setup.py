from setuptools import setup, find_packages

setup(
    name="emotion_voice_converter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "librosa>=0.9.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "soundfile>=0.10.3",
        "tensorboard>=2.8.0",
        "opensmile>=2.4.0",
        "transformers>=4.15.0",
        "ffmpeg-python>=0.2.0",
        "pesq>=0.0.3",
        "pystoi>=0.3.3",
        "tqdm>=4.62.0",
    ],
) 