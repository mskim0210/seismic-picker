from setuptools import setup, find_packages

setup(
    name="seismic_picker",
    version="0.1.0",
    description="Deep learning-based seismic phase detection and picking (TPhaseNet)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "obspy>=1.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "seisbench": ["seisbench>=0.4.0"],
    },
    entry_points={
        "console_scripts": [
            "seismic-pick=scripts.predict:main",
        ],
    },
)
