from setuptools import setup, find_packages

setup(
    name="cxgnn",
    version="0.1.0",
    description="A GNN causal explainer based on causal structure and neural causal models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",  # Add your name
    author_email="",  # Add your email
    url="https://github.com/yourusername/cxgnn",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torch_geometric>=2.0.0",
        "networkx>=2.6.3",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.2",
        "tqdm>=4.62.0",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.1",
        "rdkit>=2022.3.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cxgnn=cxgnn.main:main",
        ],
    },
)
