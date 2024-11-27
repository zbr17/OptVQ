# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from setuptools import setup, find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="optvq",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    version="0.1.0",
    description="A package for tokenization",
    author="Borui Zhang",
)