#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_dot_kernel(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) {
    // Implement the kernel to compute matrix-vector dot product
    // Each thread computes one element of the result vector
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows)
    {
        float sum = 0.f;
        for(int col = 0; col < cols; ++col)
        {
            sum += matrix[row * rows + col] * vector[col]; 
        }

        result[row] = sum;
    }
}

std::vector<float> matrix_dot_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {

    int rows = matrix.size();
    int cols = matrix[0].size();

    if(rows == 0 || cols == 0 || vec.size() != cols)
        return {-1};

    std::vector<float> flatMatrix;
    flatMatrix.reserve(rows * cols);
    for (const auto& row : matrix) 
    {
        flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
    }

    std::vector<float> result(rows, 0.0f);
    float* d_mat = nullptr, *d_vec = nullptr, *d_res = nullptr;
    cudaMalloc(&d_mat, rows * cols * sizeof(float));
    cudaMalloc(&d_vec, cols * sizeof(float));
    cudaMalloc(&d_res, rows * sizeof(float));

    cudaMemcpy(d_mat, flatMatrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    matrix_vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_vec, d_res, rows, cols);

    // 5. Transfer computed results back to Host memory
    cudaMemcpy(result.data(), d_res, rows * sizeof(float), cudaMemcpyDeviceToHost); 
    // Return empty vector if dimensions don't match
    // 1. Allocate device memory
    // 2. Copy data to device
    // 3. Launch kernel
    // 4. Copy result back
    // 5. Free memory and return result
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);
    return result;
}