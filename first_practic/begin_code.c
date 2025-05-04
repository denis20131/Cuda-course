#define USE_FLOAT
#include "common.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Конфигурация выполнения
#define BLOCK_SIZE 4
#define STREAM_COUNT 2

__global__ void update_kernel(real* A, real* B, float* d_eps, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i >= L - 1 || j >= L - 1 || k >= L - 1) return;

    int idx = IDX(i, j, k, L);
    float diff = fabsf(B[idx] - A[idx]);
    
    // Атомарное обновление максимума для float
    int* addr_as_int = (int*)d_eps;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected, 
                       __float_as_int(fmaxf(__int_as_float(expected), diff)));
    } while (expected != old);
    
    A[idx] = B[idx];
}

__global__ void compute_kernel(real* A, real* B, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i >= L - 1 || j >= L - 1 || k >= L - 1) return;

    int idx = IDX(i, j, k, L);
    B[idx] = (A[IDX(i-1,j,k,L)] + A[IDX(i+1,j,k,L)] +
              A[IDX(i,j-1,k,L)] + A[IDX(i,j+1,k,L)] +
              A[IDX(i,j,k-1,L)] + A[IDX(i,j,k+1,L)]) * (1.0f/6.0f);
}

void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("GPU Memory: %.1f GB (%.1f GB free)\n", 
           total/1024.0/1024.0/1024.0, 
           free/1024.0/1024.0/1024.0);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s L ITMAX [MAXEPS=0.5]\n", argv[0]);
        return 1;
    }

    const int L = atoi(argv[1]);
    const int ITMAX = atoi(argv[2]);
    const float MAXEPS = (argc > 3) ? atof(argv[3]) : 0.5f;
    const size_t size = L * L * L * sizeof(real);

    print_gpu_info();

    // Проверка доступной памяти
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (size * 2 > free_mem) {
        printf("Error: Not enough GPU memory available!\n");
        printf("Required: %.1f GB, Available: %.1f GB\n",
               size*2/1024.0/1024.0/1024.0,
               free_mem/1024.0/1024.0/1024.0);
        return 1;
    }

    // Выделение памяти на устройстве
    real *d_A, *d_B;
    float *d_eps;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_eps, sizeof(float));

    // Инициализация на хосте
    std::vector<real> h_A(L*L*L), h_B(L*L*L);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                int idx = IDX(i,j,k,L);
                h_A[idx] = 0;
                h_B[idx] = (i==0 || j==0 || k==0 || i==L-1 || j==L-1 || k==L-1) ? 0 : 4+i+j+k;
            }

    // Копирование на устройство
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Настройка выполнения
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((L-2 + block.x-1)/block.x, 
              (L-2 + block.y-1)/block.y,
              (L-2 + block.z-1)/block.z);

    // Создание событий для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Замер времени CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);

    // Основной цикл вычислений
    int it_done = ITMAX;
    for (int it = 0; it < ITMAX; ++it) {
        float zero = 0.0f;
        cudaMemcpy(d_eps, &zero, sizeof(float), cudaMemcpyHostToDevice);

        update_kernel<<<grid, block>>>(d_A, d_B, d_eps, L);
        compute_kernel<<<grid, block>>>(d_A, d_B, L);
        cudaDeviceSynchronize();

        // Проверка сходимости
        if (it % 10 == 0 || it == ITMAX-1) {
            float eps;
            cudaMemcpy(&eps, d_eps, sizeof(float), cudaMemcpyDeviceToHost);
            
            printf("IT = %4d EPS = %14.7E\n", it+1, eps);
            if (eps < MAXEPS) {
                it_done = it+1;
                break;
            }
        }
    }

    // Замер времени GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Замер времени CPU
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;

    // Вывод результатов
    printf("\nJacobi3D Benchmark Results\n");
    printf(" Grid Size:      %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations:     %20d\n", it_done);
    printf(" GPU Time:       %20.2f sec\n", gpu_ms/1000.0f);
    printf(" CPU Time:       %20.2f sec\n", cpu_elapsed.count());
    printf(" Performance:    %20.2f iterations/sec\n", it_done/(gpu_ms/1000.0f));
    printf(" Memory Used:    %20.1f MB\n", size*2/1024.0/1024.0);

    // Освобождение ресурсов
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_eps);
    
    return 0;
}