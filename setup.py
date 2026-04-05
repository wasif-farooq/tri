from setuptools import setup, find_packages

setup(
    name="torch-range-indexed",
    version="1.0.0",
    author="Your Name",
    description="Run large language models on small GPUs with zero accuracy loss",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gpu": [
            "cuda-python>=11.7",  # For CUDA kernel compilation
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
    },
    # Include CUDA files in package
    package_data={
        "tri.gpu": ["*.cu"],
    },
)