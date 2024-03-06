from setuptools import setup, find_packages
from pathlib import Path

import os

dir_path = Path('__file__').parent.absolute()
md_path = os.path.join(dir_path, "README.md")
long_description = Path(md_path).read_text(encoding='utf-8')

setup(
    name="drug_synergy_pred",
    version="1.0.0",
    description="Drug Synergy Prediction based on TDC Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidanugraha/solubilitypred",
    author="David Anugraha",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry :: Chemoinformatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ML, drug synergy",
    packages=find_packages(),
    python_requires=">=3.10.12, <4",
    install_requires=[
        "matplotlib>=3.8.3",
        "mordred>=1.2.0",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "rdkit>=2023.9.5",
        "scikit-learn>=1.4.1",
        "scipy>=1.12.0",
        "torch>=2.2.1",
        "torch_geometric>=2.5.0",
        "tqdm>=4.66.2",
    ],
    project_urls={
        "Bug Reports": "https://github.com/davidanugraha/drug_synergy_pred/issues",
        "Source": "https://github.com/davidanugraha/drug_synergy_pred",
    },
)