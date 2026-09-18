import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplog",
    version="1.0.0",
    author="DeepLog Research & Engineering Contributors",
    description="PyTorch implementation of DeepLog: Log Anomaly Detection and Sequence Diagnosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "torch-train>=0.0.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
)
