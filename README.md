

# CUDA GPU Image Processing

A GPU-accelerated image processing project developed using CUDA C++ and the NVIDIA Performance Primitives (NPP) library.

The project processes a batch of real-world images using GPU computation and records execution performance.

## Dataset

This project uses the **USC SIPI Image Database, Volume 3: Miscellaneous** dataset.

Official source:

https://sipi.usc.edu/database/database.php?image=26&volume=misc

The dataset contains grayscale and color images at multiple resolutions. The TIFF images are converted to grayscale PGM format before CUDA processing.

### Dataset preparation

The original SIPI TIFF files are not stored in this repository.

Download the dataset:

```text
https://sipi.usc.edu/database/misc.zip

Extract it into:

sipi_dataset/misc/

Install Pillow:

python -m pip install Pillow

Convert the TIFF images to PGM:

python prepare_sipi_dataset.py

The converted images are placed in:

input/
Project Structure
CUDA-GPU-Image-Processing/
│
├── src/
│   ├── gpu_processing.cu
│   └── image_processing.cu
│
├── input/
│   └── SIPI PGM images
│
├── output/
│   └── processed images
│
├── results/
│   ├── performance.csv
│   └── execution.log
│
├── screenshots/
│
├── presentation/
│
├── prepare_sipi_dataset.py
├── run.ps1
├── README.md
└── .gitignore
GPU Processing

The project supports GPU-based image processing operations including:

5x5 Gaussian/box-style blur using NVIDIA NPP
Contrast enhancement using a custom CUDA kernel
Batch processing of multiple images
CUDA event-based execution timing
CUDA Kernel

The custom contrast operation uses a CUDA kernel with:

2D thread blocks
Global device memory
Grid and block indexing
Boundary checking
Parallel pixel processing
NVIDIA NPP

The blur operation uses the NVIDIA Performance Primitives library:

nppiFilterBoxBorder_8u_C1R

This performs GPU-accelerated filtering on grayscale image data.

Requirements

The execution environment requires:

NVIDIA GPU
CUDA Toolkit
CUDA compiler (nvcc)
NVIDIA NPP library
C++17 compatible compiler

Python is required only for dataset preparation.

Build

On a CUDA-enabled Linux environment:

nvcc -std=c++17 \
    src/image_processing.cu \
    src/gpu_processing.cu \
    -o gpu_image_processor \
    -lnppif -lnppc
Run

Blur processing:

./gpu_image_processor \
    --input input \
    --output output \
    --operation blur \
    --iterations 10

Contrast processing:

./gpu_image_processor \
    --input input \
    --output output \
    --operation contrast \
    --iterations 10
Performance Results

Execution results are stored in:

results/performance.csv
results/execution.log

The performance measurements include:

Image name
Image dimensions
Processing operation
Number of iterations
GPU processing time
Total processing time

Final GPU performance values will be recorded after execution on an NVIDIA CUDA-enabled environment.

Output

Processed images are written to:

output/

Execution evidence and screenshots are stored in:

screenshots/

The final project presentation is stored in:

presentation/
Project Objective

The objective is to demonstrate how CUDA GPU parallelism can accelerate image processing across a batch of images compared with traditional CPU-based processing.

The project demonstrates:

GPU memory management
Host-to-device memory transfers
CUDA kernel execution
NVIDIA NPP image processing
Batch image processing
CUDA event timing
GPU performance measurement
Author

Ankita Birajdar

Computer Science & Engineering


Save the file with:
