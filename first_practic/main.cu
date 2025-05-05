#include <stdio.h>
#include "common.hpp"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>

#define BLOCK_SIZE 8


__device__ float atomicMaxFloat(float* address, float val) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                      __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void jacobi_kernel(real* A, real* B, float* d_eps, int L, int z_start, int z_end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + z_start + 1;

    if (i >= L-1 || j >= L-1 || k >= z_end) return;

    int idx = IDX(i,j,k,L);
    real new_val = (A[IDX(i-1,j,k,L)] + A[IDX(i+1,j,k,L)] +
                   A[IDX(i,j-1,k,L)] + A[IDX(i,j+1,k,L)] +
                   A[IDX(i,j,k-1,L)] + A[IDX(i,j,k+1,L)]) * (1.0f/6.0f);
    
    float diff = fabsf(new_val - A[idx]);
    
    __shared__ float block_max;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        block_max = 0.0f;
    }
    __syncthreads();

    atomicMaxFloat(&block_max, diff);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicMaxFloat(d_eps, block_max);
    }
    
    B[idx] = new_val;
}

void print_memory_info(int dev) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    printf("GPU %d: %s\n", dev, prop.name);
    printf("  Total memory: %.2f MB\n", total/1024.0/1024.0);
    printf("  Free memory:  %.2f MB\n", free/1024.0/1024.0);
    printf("  Used memory:  %.2f MB\n", (total-free)/1024.0/1024.0);
}

void initialize_data(real* h_A, real* h_B, int L) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int idx = IDX(i,j,k,L);
                h_A[idx] = 0.0f;
                if (i == 0 || j == 0 || k == 0 || i == L-1 || j == L-1 || k == L-1) {
                    h_B[idx] = 0.0f;
                } else {
                    h_B[idx] = (float)(i + j + k);
                }
            }
        }
    }
}

void run_on_gpu(int dev, real* d_A, real* d_B, float* d_eps, int L,int MAX_IT,const float MAXEPS) {
    cudaSetDevice(dev);
    
    int z_size = (L-2 + MAX_GPUS - 1) / MAX_GPUS;
    int z_start = dev * z_size;
    int z_end = min(z_start + z_size, L-2);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((L-2 + block.x-1)/block.x,
              (L-2 + block.y-1)/block.y,
              (z_end - z_start + block.z-1)/block.z);

    // Выводим информацию о памяти перед выполнением
    printf("\nMemory usage before computation (GPU %d):\n", dev);
    print_memory_info(dev);

    for (int iter = 0; iter < MAX_IT; iter++) {
        float zero = 0.0f;
        cudaMemcpy(d_eps, &zero, sizeof(float), cudaMemcpyHostToDevice);
        
        jacobi_kernel<<<grid, block>>>(d_A, d_B, d_eps, L, z_start, z_end);
        cudaDeviceSynchronize();
        
        real* temp = d_A;
        d_A = d_B;
        d_B = temp;
        
        float eps;
        cudaMemcpy(&eps, d_eps, sizeof(float), cudaMemcpyDeviceToHost);
	if (iter==0) continue;
        printf("GPU %d Iter %2d EPS = %.3e\n", dev, iter+1, eps);
        printf("GPU %d Iter %2d EPS = %.3e\n", dev, iter+1, eps);
        if (eps < MAXEPS) break;
    }

    // Выводим информацию о памяти после выполнения
    printf("\nMemory usage after computation (GPU %d):\n", dev);
    print_memory_info(dev);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <grid_size> <max_iterations> [use_float=1]\n", argv[0]);
        return 1;
    }
    const float MAXEPS = 0;
    const int L = atoi(argv[1]);
    const int ITMAX = atoi(argv[2]);
    const size_t mem_size = L*L*L*sizeof(real);
    const size_t eps_size = sizeof(float);
    
    printf("Problem size: %dx%dx%d\n", L, L, L);
    printf("Memory per array: %.2f MB\n", mem_size/1024.0/1024.0);
    printf("Total GPU memory required: %.2f MB (per GPU)\n", 
          (2*mem_size + eps_size)/1024.0/1024.0);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices < MAX_GPUS) {
        printf("Error: Need at least %d GPUs (found %d)\n", MAX_GPUS, num_devices);
        return 1;
    }

    // Выводим информацию о доступных GPU
    for (int dev = 0; dev < MAX_GPUS; dev++) {
        cudaSetDevice(dev);
        printf("\nGPU %d properties:\n", dev);
        print_memory_info(dev);
    }

    // Инициализация данных
    real *h_A = (real*)malloc(mem_size);
    real *h_B = (real*)malloc(mem_size);
    initialize_data(h_A, h_B, L);

    // Выделение памяти на устройствах
    real *d_A[MAX_GPUS], *d_B[MAX_GPUS];
    float *d_eps[MAX_GPUS];
    
    for (int dev = 0; dev < MAX_GPUS; dev++) {
        cudaSetDevice(dev);
        cudaMalloc(&d_A[dev], mem_size);
        cudaMalloc(&d_B[dev], mem_size);
        cudaMalloc(&d_eps[dev], eps_size);
        
        cudaMemcpy(d_A[dev], h_A, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B[dev], h_B, mem_size, cudaMemcpyHostToDevice);
    }

    // Запуск
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for num_threads(MAX_GPUS)
    for (int dev = 0; dev < MAX_GPUS; dev++) {
        run_on_gpu(dev, d_A[dev], d_B[dev], d_eps[dev], L,ITMAX,MAXEPS);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end-start).count();
    printf("\nTotal computation time: %.3f seconds\n", time);

    // Освобождение памяти
    free(h_A);
    free(h_B);
    for (int dev = 0; dev < MAX_GPUS; dev++) {
        cudaSetDevice(dev);
        cudaFree(d_A[dev]);
        cudaFree(d_B[dev]);
        cudaFree(d_eps[dev]);
    }

    return 0;
}