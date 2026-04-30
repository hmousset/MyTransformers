import io
import os
from typing import List

import setuptools

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


# Core deps — always required, lightweight enough for any environment.
CORE_REQUIREMENTS = [
    "torch",
    "numpy",
    "pytz",
    "scikit-learn",
    "sentencepiece",
    "transformers",
]

# Optional heavy deps — needed for specific training modes.
EXTRAS = {
    "deepspeed": ["deepspeed", "fairscale==0.4.13"],
    "vllm": ["vllm"],
    "liger": ["liger_kernel"],
    "quant": ["bitsandbytes"],
    "logging": ["wandb"],
    "pca": ["torch_incremental_pca"],
    "blob": ["blobfile"],
    "full": [
        "deepspeed", "fairscale==0.4.13", "vllm", "liger_kernel",
        "bitsandbytes", "wandb", "torch_incremental_pca", "blobfile",
    ],
}


setuptools.setup(
    name="MyTransformers",
    version="0.1",
    author="haonan he, yucheng ren",
    description=("MyTransformers library implementation"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("benchmarks", "docs",
                                               "examples", "tests")),
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS,
)
