#include <cuda_runtime.h>

__global__ void contrastKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const int index = y * width + x;

    const int value = input[index];

    int adjusted = ((value - 128) * 3) / 2 + 128;

    if (adjusted < 0)
    {
        adjusted = 0;
    }

    if (adjusted > 255)
    {
        adjusted = 255;
    }

    output[index] =
        static_cast<unsigned char>(adjusted);
}

void LaunchContrastKernel(
    const unsigned char* deviceInput,
    unsigned char* deviceOutput,
    int width,
    int height)
{
    const dim3 blockSize(16, 16);

    const dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    contrastKernel<<<gridSize, blockSize>>>(
        deviceInput,
        deviceOutput,
        width,
        height);
}