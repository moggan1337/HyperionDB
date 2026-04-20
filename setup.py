"""
HyperionDB Setup Script
======================

Self-Driving Database with Learned Optimizer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperiondb",
    version="1.0.0",
    author="HyperionDB Team",
    author_email="team@hyperiondb.dev",
    description="Self-Driving Database with Learned Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/HyperionDB",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "torch": [
            "torch>=2.0.0",
        ],
        "tensorflow": [
            "tensorflow>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hyperiondb=hyperiondb.core.database:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
