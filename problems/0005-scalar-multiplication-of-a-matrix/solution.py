#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void scalar_multiply_kernel(
    const float* input,
    float* output,
    float scalar,
    int total_elements
) 
{
    // Implement the kernel to multiply each element by scalar
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < total_elements)
    {
        output[workIndex] = input[workIndex] * scalar;
    }
}

std::vector<std::vector<float>> scalar_multiply(const std::vector<std::vector<float>>& matrix, float scalar) {
    // 1. Allocate device memory
    // 2. Copy data to device
    // 3. Launch kernel
    // 4. Copy result back
    // 5. Free memory and return result
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<float> flatarray(rows * cols);
    for (size_t r = 0; r < rows; ++r)
    {
        if (matrix[r].size() != cols)
            return {};
        std::copy(matrix[r].begin(), matrix[r].end(), flatarray.begin() + r * cols);
    }
    float* input = nullptr, *output = nullptr;
    cudaMalloc(&input, rows * cols * sizeof(float));
    cudaMalloc(&output, rows * cols * sizeof(float));

    cudaMemcpy(input, flatarray.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    size_t blocks = static_cast<size_t>((rows * cols + threads - 1) / threads);
    scalar_multiply_kernel <<<threads, blocks>>> (input, output, scalar, rows * cols);
    cudaDeviceSynchronize();

    cudaMemcpy(flatarray.data(), output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(output);

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    for (size_t r = 0; r < rows; ++r)
    {
        std::copy(flatarray.begin() + r * cols, flatarray.begin() + (r + 1) * cols, result[r].begin());
    }

    return result;
}