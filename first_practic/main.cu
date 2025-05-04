#define USE_FLOAT
#include "common.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Конфигурация выполнения
#define BLOCK_SIZE 8

// Атомарный максимум для float
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

__global__ void update_kernel(real* A, real* B, float* d_eps, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i >= L - 1 || j >= L - 1 || k >= L - 1) return;

    int idx = IDX(i, j, k, L);
    float diff = fabsf(B[idx] - A[idx]);
    
    // Редукция внутри блока через shared memory
    __shared__ float shared_max[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
    int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    shared_max[tid] = diff;
    __syncthreads();

    for (int s = blockDim.x*blockDim.y*blockDim.z/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Атомарное обновление глобального максимума
    if (tid == 0) {
        atomicMaxFloat(d_eps, shared_max[0]);
    }
    
    A[idx] = B[idx];
}

__global__ void compute_kernel(real* A, real* B, int L) {
    __shared__ real tile[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2]; // +2 для halo
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Загрузка блока в shared memory (с halo)
    if(i >= 1 && i <= L-2 && j >= 1 && j <= L-2 && k >= 1 && k <= L-2) {
        tile[threadIdx.x][threadIdx.y][threadIdx.z] = A[IDX(i,j,k,L)];
        
        // Загрузка halo-областей
        if(threadIdx.x == 0) tile[threadIdx.x-1][threadIdx.y][threadIdx.z] = A[IDX(i-1,j,k,L)];
        if(threadIdx.x == blockDim.x-1) tile[threadIdx.x+1][threadIdx.y][threadIdx.z] = A[IDX(i+1,j,k,L)];
        // ... аналогично для y и z
    }
    __syncthreads();

    if(i >= 1 && i <= L-2 && j >= 1 && j <= L-2 && k >= 1 && k <= L-2) {
        B[IDX(i,j,k,L)] = (tile[threadIdx.x-1][threadIdx.y][threadIdx.z] +
                          tile[threadIdx.x+1][threadIdx.y][threadIdx.z] +
                          tile[threadIdx.x][threadIdx.y-1][threadIdx.z] +
                          tile[threadIdx.x][threadIdx.y+1][threadIdx.z] +
                          tile[threadIdx.x][threadIdx.y][threadIdx.z-1] +
                          tile[threadIdx.x][threadIdx.y][threadIdx.z+1]) * (1.0f/6.0f);
    }
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

    // Проверка памяти
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (size * 2 > free_mem) {
        printf("Error: Not enough GPU memory! Required %.1f GB, available %.1f GB\n",
               size*2/1024.0/1024.0/1024.0, free_mem/1024.0/1024.0/1024.0);
        return 1;
    }

    // Выделение памяти
    real *d_A, *d_B;
    float *d_eps;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_eps, sizeof(float));

    // Инициализация
    std::vector<real> h_A(L*L*L), h_B(L*L*L);
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                int idx = IDX(i,j,k,L);
                h_A[idx] = 0;
                h_B[idx] = (i==0 || j==0 || k==0 || i==L-1 || j==L-1 || k==L-1) ? 0 : 4+i+j+k;
            }
        }
    }
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Настройка выполнения
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((L-2 + block.x-1)/block.x, 
              (L-2 + block.y-1)/block.y,
              (L-2 + block.z-1)/block.z);

    // Измерение времени
    auto start = std::chrono::high_resolution_clock::now();
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    // Основной цикл
    int it_done = 0;
    for (int it = 0; it < ITMAX; ++it) {
        float zero = 0.0f;
        cudaMemcpy(d_eps, &zero, sizeof(float), cudaMemcpyHostToDevice);

        update_kernel<<<grid, block>>>(d_A, d_B, d_eps, L);
        compute_kernel<<<grid, block>>>(d_A, d_B, L);
        cudaDeviceSynchronize();

        float eps;
        cudaMemcpy(&eps, d_eps, sizeof(float), cudaMemcpyDeviceToHost);
        printf("IT = %4d EPS = %14.7E\n", it+1, eps);

        if (eps < MAXEPS) {
            it_done = it+1;
            break;
        }
    }
    it_done = (it_done == 0) ? ITMAX : it_done;

    // Замер времени
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Результаты
    printf("\nJacobi3D Benchmark Results\n");
    printf(" Grid Size:      %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations:     %20d\n", it_done);
    printf(" GPU Time:       %20.2f sec\n", gpu_ms/1000.0f);
    printf(" Performance:    %20.2f iters/sec\n", it_done/(gpu_ms/1000.0f));
    printf(" Memory Used:    %20.1f MB\n", size*2/1024.0/1024.0);


    return 0;
}