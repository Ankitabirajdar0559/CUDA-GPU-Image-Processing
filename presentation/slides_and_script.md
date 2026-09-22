# Project Presentation & Demo Script (5-10 Minutes)

## Spoken Script (Word-for-Word)
Hello everyone. Welcome to my final project presentation for CUDA at Scale for the Enterprise. My name is Ankita Birajdar, and my project focuses on accelerating batch image processing using CUDA C++ and the NVIDIA Performance Primitives (NPP) library.

For this project, I utilized the USC SIPI miscellaneous image database, converting TIFF images to PGM format for raw memory layout processing.

I implemented two primary operations:
1. A custom CUDA 2D kernel for parallel contrast adjustment, where threads are mapped using 2D grid dimensions (16x16 threads per block).
2. NVIDIA NPP library integration using nppiFilterBoxBorder_8u_C1R for optimized 5x5 spatial blurring.

As recorded in results/performance.csv, the GPU shows significant speedups as resolution grows: from 9x on 256x256 images to roughly 25x on 1024x1024 images over single-threaded CPU execution. CUDA events were used to measure pure kernel time versus PCIe memory transfer overhead.

All outputs were verified against CPU reference runs with zero discrepancies. Thank you!
