#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) 
{
    // Implement the kernel for matrix multiplication C = A * B
    // A is M x K, B is K x N, C is M x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

std::vector<std::vector<float>> matrixmul(
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b
) {
    int arows = a.size();
    int acols = a[0].size();

    int brows = b.size();
    int bcols = b[0].size();

    if(arows == 0 || acols == 0 || brows == 0 || bcols == 0 || acols != brows)
        return {{-1}};

    std::vector<float> aflat(arows * acols);
    for(int i = 0; i < arows; ++i)
    {
        if(a[i].size() != acols)
            return {{-1}};
        std::copy(a[i].begin(), a[i].end(), aflat.begin() + i * acols);
    }

    std::vector<float> bflat(brows * bcols);
    for(int i = 0; i < brows; ++i)
    {
        if(b[i].size() != bcols)
            return {{-1}};
        std::copy(b[i].begin(), b[i].end(), bflat.begin() + i * bcols);
    }
    // Return {-1} if dimensions don't align
    // 1. Allocate device memory
    // 2. Copy data to device
    // 3. Launch kernel
    // 4. Copy result back
    // 5. Free memory and return result

    dim3 block(16, 16);

    float* d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, arows * acols * sizeof(float));
    cudaMalloc(&d_B, brows * bcols * sizeof(float));
    cudaMalloc(&d_C, arows * bcols * sizeof(float));

    cudaMemcpy(d_A, aflat.data(), arows * acols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, bflat.data(), brows * bcols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((bcols + block.x - 1) / block.x, (arows + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, arows, acols, bcols);
    cudaDeviceSynchronize();
    std::vector<float> flatC (arows * bcols);
    cudaMemcpy(flatC.data(), d_C, arows * bcols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::vector<std::vector<float>> result(arows,std::vector<float>(bcols));

    for (int i = 0; i < arows; ++i)
    {
        std::copy(flatC.begin() + i * bcols, flatC.begin() + (i + 1) * bcols, result[i].begin());
    }
    return result;
}