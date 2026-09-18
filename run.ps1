param(
    [string]$Operation = "blur",
    [int]$Iterations = 10
)

Write-Host "========================================"
Write-Host "CUDA GPU Image Processing"
Write-Host "========================================"
Write-Host ""

Write-Host "Building project..."

nvcc -std=c++17 `
    src/image_processing.cu `
    src/gpu_processing.cu `
    -o gpu_image_processor.exe `
    -lnppif -lnppc

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed."
    exit 1
}

Write-Host ""
Write-Host "Build successful."
Write-Host ""

Write-Host "Running operation: $Operation"
Write-Host "Iterations: $Iterations"
Write-Host ""

.\gpu_image_processor.exe `
    --input input `
    --output output `
    --operation $Operation `
    --iterations $Iterations