#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_filter_functions.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void LaunchContrastKernel(
    const unsigned char* deviceInput,
    unsigned char* deviceOutput,
    int width,
    int height);

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess)                             \
        {                                                     \
            std::cerr << "CUDA error: "                          \
                      << cudaGetErrorString(error)            \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << "\n";                                \
            return false;                                     \
        }                                                     \
    } while (0)

#define NPP_CHECK(call)                                      \
    do                                                       \
    {                                                        \
        NppStatus status = (call);                           \
        if (status != NPP_SUCCESS)                           \
        {                                                    \
            std::cerr << "NPP error: " << status             \
                      << " at " << __FILE__ << ":"           \
                      << __LINE__ << "\n";                   \
            return false;                                    \
        }                                                    \
    } while (0)

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

bool ReadPGM(const fs::path& filename, Image& image)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cerr << "Unable to open: "
                  << filename << "\n";
        return false;
    }

    std::string magic;
    file >> magic;

    if (magic != "P5")
    {
        std::cerr << "Only binary PGM P5 files are supported.\n";
        return false;
    }

    file >> image.width;
    file >> image.height;

    int maxValue;
    file >> maxValue;

    if (image.width <= 0 ||
        image.height <= 0 ||
        maxValue != 255)
    {
        std::cerr << "Invalid PGM format.\n";
        return false;
    }

    file.get();

    const size_t imageBytes =
        static_cast<size_t>(image.width) *
        static_cast<size_t>(image.height);

    image.pixels.resize(imageBytes);

    file.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<std::streamsize>(imageBytes));

    if (!file)
    {
        std::cerr << "Failed to read image data.\n";
        return false;
    }

    return true;
}

bool WritePGM(
    const fs::path& filename,
    const Image& image)
{
    std::ofstream file(
        filename,
        std::ios::binary);

    if (!file)
    {
        std::cerr << "Unable to write: "
                  << filename << "\n";
        return false;
    }

    file << "P5\n";
    file << image.width << " "
         << image.height << "\n";
    file << "255\n";

    file.write(
        reinterpret_cast<const char*>(
            image.pixels.data()),
        static_cast<std::streamsize>(
            image.pixels.size()));

    return static_cast<bool>(file);
}

bool ProcessImage(
    const Image& inputImage,
    Image& outputImage,
    const std::string& operation,
    int iterations,
    float& gpuTimeMs)
{
    outputImage.width = inputImage.width;
    outputImage.height = inputImage.height;

    outputImage.pixels.resize(
        inputImage.pixels.size());

    const size_t imageBytes =
        inputImage.pixels.size();

    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    CUDA_CHECK(cudaMalloc(
        &deviceInput,
        imageBytes));

    CUDA_CHECK(cudaMalloc(
        &deviceOutput,
        imageBytes));

    CUDA_CHECK(cudaMemcpy(
        deviceInput,
        inputImage.pixels.data(),
        imageBytes,
        cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        if (operation == "blur")
        {
            NppiSize sourceSize{
                inputImage.width,
                inputImage.height
            };

            NppiPoint sourceOffset{
                0,
                0
            };

            NppiSize roiSize{
                inputImage.width,
                inputImage.height
            };

            NppiSize maskSize{
                5,
                5
            };

            NppiPoint anchor{
                2,
                2
            };

            NPP_CHECK(
                nppiFilterBoxBorder_8u_C1R(
                    deviceInput,
                    inputImage.width,
                    sourceSize,
                    sourceOffset,
                    deviceOutput,
                    inputImage.width,
                    roiSize,
                    maskSize,
                    anchor,
                    NPP_BORDER_REPLICATE));
        }
        else if (operation == "contrast")
        {
            LaunchContrastKernel(
                deviceInput,
                deviceOutput,
                inputImage.width,
                inputImage.height);

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(
            deviceInput,
            deviceOutput);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(
        &gpuTimeMs,
        start,
        stop));

    CUDA_CHECK(cudaMemcpy(
        outputImage.pixels.data(),
        deviceInput,
        imageBytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));

    return true;
}

void PrintUsage()
{
    std::cout
        << "\nUsage:\n"
        << "  gpu_image_processor "
        << "--input <folder> "
        << "--output <folder> "
        << "--operation <blur|contrast> "
        << "--iterations <number>\n\n";
}

int main(int argc, char* argv[])
{
    std::string inputDirectory = "input";
    std::string outputDirectory = "output";
    std::string operation = "blur";
    int iterations = 10;

    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];

        if (argument == "--input" &&
            i + 1 < argc)
        {
            inputDirectory = argv[++i];
        }
        else if (argument == "--output" &&
                 i + 1 < argc)
        {
            outputDirectory = argv[++i];
        }
        else if (argument == "--operation" &&
                 i + 1 < argc)
        {
            operation = argv[++i];
        }
        else if (argument == "--iterations" &&
                 i + 1 < argc)
        {
            iterations =
                std::stoi(argv[++i]);
        }
        else if (argument == "--help")
        {
            PrintUsage();
            return 0;
        }
        else
        {
            std::cerr
                << "Unknown argument: "
                << argument << "\n";

            PrintUsage();
            return 1;
        }
    }

    if (operation != "blur" &&
        operation != "contrast")
    {
        std::cerr
            << "Operation must be "
            << "blur or contrast.\n";

        return 1;
    }

    if (iterations <= 0)
    {
        std::cerr
            << "Iterations must be greater than zero.\n";

        return 1;
    }

    fs::create_directories(outputDirectory);
    fs::create_directories("results");

    int deviceCount = 0;

    CUDA_CHECK(cudaGetDeviceCount(
        &deviceCount));

    if (deviceCount == 0)
    {
        std::cerr
            << "No CUDA GPU detected.\n";

        return 1;
    }

    cudaDeviceProp deviceProperties{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &deviceProperties,
        0));

    std::cout
        << "CUDA GPU: "
        << deviceProperties.name
        << "\n";

    std::cout
        << "Input directory: "
        << inputDirectory
        << "\n";

    std::cout
        << "Output directory: "
        << outputDirectory
        << "\n";

    std::cout
        << "Operation: "
        << operation
        << "\n";

    std::cout
        << "Iterations: "
        << iterations
        << "\n\n";

    const fs::path resultsFile =
        "results/performance.csv";

    std::ofstream csv(resultsFile);

    csv << "image,width,height,operation,"
           "gpu_time_ms\n";

    int processedImages = 0;

    for (const auto& entry :
         fs::directory_iterator(inputDirectory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (entry.path().extension() != ".pgm")
        {
            continue;
        }

        Image inputImage;
        Image outputImage;

        if (!ReadPGM(
                entry.path(),
                inputImage))
        {
            continue;
        }

        float gpuTimeMs = 0.0f;

        if (!ProcessImage(
                inputImage,
                outputImage,
                operation,
                iterations,
                gpuTimeMs))
        {
            return 1;
        }

        const fs::path outputFile =
            fs::path(outputDirectory) /
            entry.path().filename();

        if (!WritePGM(
                outputFile,
                outputImage))
        {
            return 1;
        }

        csv << entry.path().stem().string()
            << ".pgm,"
            << inputImage.width
            << ","
            << inputImage.height
            << ","
            << operation
            << ","
            << std::fixed
            << std::setprecision(4)
            << gpuTimeMs
            << "\n";

        std::cout
            << "Processed "
            << entry.path().filename().string()
            << " | GPU time: "
            << std::fixed
            << std::setprecision(4)
            << gpuTimeMs
            << " ms\n";

        ++processedImages;
    }

    csv.close();

    std::ofstream log(
        "results/execution.log");

    log << "CUDA GPU Image Processing Capstone\n";
    log << "GPU: "
        << deviceProperties.name
        << "\n";
    log << "Operation: "
        << operation
        << "\n";
    log << "Iterations: "
        << iterations
        << "\n";
    log << "Images processed: "
        << processedImages
        << "\n";
    log << "Results: "
        << resultsFile.string()
        << "\n";

    log.close();

    std::cout
        << "\nProcessing completed successfully.\n";

    std::cout
        << "Images processed: "
        << processedImages
        << "\n";

    std::cout
        << "Performance: "
        << resultsFile.string()
        << "\n";

    return 0;
}