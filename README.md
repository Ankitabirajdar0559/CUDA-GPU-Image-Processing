# CUDA GPU Image Processing and Performance Analysis

## Project Overview

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA and the CUDA NPP library.

The application processes a dataset containing 20 grayscale images, each with a resolution of 1024 x 1024 pixels.

Two GPU processing techniques are implemented:

1. NPP 5x5 Box Filter for image blurring.
2. A custom CUDA kernel for image contrast enhancement.

GPU execution time is measured using CUDA events and recorded in CSV format.

## Technologies

- C++
- CUDA
- CUDA NPP
- NVIDIA GPU
- Python
- PGM image format

## GPU Requirement

The project requires an NVIDIA CUDA-capable GPU.

The development environment used for testing was:

- NVIDIA L4 GPU
- CUDA Toolkit
- CUDA NPP

## Dataset

The project generates 20 grayscale PGM images.

Each image:

- Resolution: 1024 x 1024
- Color format: grayscale
- Pixel depth: 8-bit
- Format: PGM P5

The dataset contains approximately 20 million pixels and satisfies the requirement for tens of large inputs.

## Project Structure

```text
CUDA-GPU-Image-Processing/
│
├── src/
│   ├── image_processing.cu
│   └── gpu_processing.cu
│
├── input/
├── output/
├── results/
├── screenshots/
├── presentation/
│
├── generate_images.py
├── run.ps1
├── README.md
└── .gitignore