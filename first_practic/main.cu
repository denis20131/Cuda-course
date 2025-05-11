#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>

#define MAX_EPS 0.5
#define GRID_LEN 900
#define MAX_ITERS 20
#define BLOCK_X 32
#define BLOCK_Y 4
#define BLOCK_Z 4

#define INDEX(x, y, z, N) (((z) * (N) * (N)) + ((y) * (N)) + (x))

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void calcEpsAndCompare(const double* oldGrid, const double* newGrid, double* diff, int N) {

    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    double threadMax = 0.0;
    
    if (x < N-1 && y < N-1 && z < N-1) {
        int idx = INDEX(x, y, z, N);
        threadMax = fabs(newGrid[idx] - oldGrid[idx]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        threadMax = fmax(threadMax, __shfl_down_sync(0xFFFFFFFF, threadMax, offset));
    }

    __shared__ double blockMax[32];  
    if (threadIdx.x % 32 == 0) {
        blockMax[threadIdx.x / 32] = threadMax;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < blockDim.x * blockDim.y * blockDim.z / 32) ? 
                    blockMax[threadIdx.x] : 0.0;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        
        if (threadIdx.x == 0) {
            atomicMaxDouble(diff, val);
        }
    }
}

__global__ void jacobiStep(const double* input, double* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x < N - 1 && y < N - 1 && z < N - 1) {
        int idx = INDEX(x, y, z, N);
        output[idx] = (input[idx - N * N] + input[idx + N * N] +
                       input[idx - N] + input[idx + N] +
                       input[idx - 1] + input[idx + 1]) / 6.0;
    }
}


void fillGrids(double* A, double* B, int N) {
    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int idx = INDEX(x, y, z, N);
                A[idx] = 0;
                B[idx] = (x == 0 || y == 0 || z == 0 || x == N - 1 || y == N - 1 || z == N - 1)
                         ? 0 : 4 + x + y + z;
            }
        }
    }
}

int main() {
    const int N = GRID_LEN;
    const size_t gridSizeBytes = N * N * N * sizeof(double);
    const size_t scalarSize = sizeof(double);

    printf("Initializing CUDA...\n");

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Device: %s | Compute Capability: %d.%d\n", devProp.name, devProp.major, devProp.minor);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("Memory: %.2f GB total | %.2f GB free\n", totalMem / 1e9, freeMem / 1e9);
    printf("Grid: %d x %d x %d | Memory per array: %.2f MB\n", N, N, N, gridSizeBytes / (1024.0 * 1024.0));

    double *hostOld = (double*)malloc(gridSizeBytes);
    double *hostNew = (double*)malloc(gridSizeBytes);
    fillGrids(hostOld, hostNew, N);

    double *devOld, *devNew, *devDiff;
    cudaMalloc(&devOld, gridSizeBytes);
    cudaMalloc(&devNew, gridSizeBytes);
    cudaMalloc(&devDiff, scalarSize);

    cudaMemcpy(devOld, hostOld, gridSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devNew, hostNew, gridSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 gridDim((N + BLOCK_X - 1) / BLOCK_X,
                 (N + BLOCK_Y - 1) / BLOCK_Y,
                 (N + BLOCK_Z - 1) / BLOCK_Z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    double* devA = devOld;
    double* devB = devNew;
    
    for (int iter = 1; iter <= MAX_ITERS; iter++) {
        double zero = 0.0;
        cudaMemcpy(devDiff, &zero, sizeof(double), cudaMemcpyHostToDevice);
    
        calcEpsAndCompare<<<gridDim, blockDim>>>(devA, devB, devDiff, N);
        jacobiStep<<<gridDim, blockDim>>>(devB, devA, N);
    
        double maxDiff = 0.0;
        cudaMemcpy(&maxDiff, devDiff, sizeof(double), cudaMemcpyDeviceToHost);
    
        printf("Iter %3d | Max Eps = %.7E\n", iter, maxDiff);
        if (maxDiff < MAX_EPS) break;
    
        double* temp = devA;
        devA = devB;
        devB = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeMs = 0.0f;
    cudaEventElapsedTime(&timeMs, start, stop);

    printf("\n=== Jacobi 3D Report ===\n");
    printf("Grid Size: %d^3\n", N);
    printf("Max Iterations: %d\n", MAX_ITERS);
    printf("Elapsed Time: %.3f sec\n", timeMs / 1000.0f);
    printf("Iterations/sec: %.2f\n", MAX_ITERS / (timeMs / 1000.0f));
    printf("Memory Used: %.2f MB\n", (2 * gridSizeBytes + scalarSize) / (1024.0 * 1024.0));

    cudaFree(devOld);
    cudaFree(devNew);
    cudaFree(devDiff);
    free(hostOld);
    free(hostNew);

    return 0;
}
