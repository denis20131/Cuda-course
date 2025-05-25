#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define GRID_X 900
#define GRID_Y 900
#define GRID_Z 900
#define MAX_ITERATIONS 10
#define CONVERGENCE_THRESHOLD 0.01

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ double gpu_atomic_max(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double current_val = __longlong_as_double(assumed);
        if (val <= current_val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void process_x_axis(double* field, int size_x, int size_y, int size_z) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= 1 && y < size_y-1 && z >= 1 && z < size_z-1) {
        for (int x = 1; x < size_x-1; x++) {
            const int idx = x*size_y*size_z + y*size_z + z;
            field[idx] = (field[idx - size_y*size_z] + field[idx + size_y*size_z]) * 0.5;
        }
    }
}

__global__ void process_y_axis(double* field, int size_x, int size_y, int size_z) {
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= 1 && x < size_x-1 && z >= 1 && z < size_z-1) {
        for (int y = 1; y < size_y-1; y++) {
            const int idx = x*size_y*size_z + y*size_z + z;
            field[idx] = (field[idx - size_z] + field[idx + size_z]) * 0.5;
        }
    }
}

__global__ void rearrange_data(double* a, double* b, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            int idx_a = k * ny * nz + i * nz + j;
            int idx_b = j * nx * ny + k * ny + i;
            b[idx_b] = a[idx_a];
        }
    }
}

__global__ void process_z_axis(double* grid, int dimx, int dimy, int dimz, double* max_error_ptr) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    bool inside = (row > 0) & (row < dimx - 1) & 
                  (col > 0) & (col < dimy - 1);

    double my_max_diff = 0.0;

    if (inside) {
        for (int depth = 1; depth < dimz - 1; depth++) {
            int cell = depth * dimx * dimy + row * dimy + col;
            int upper = cell - dimx * dimy;
            int lower = cell + dimx * dimy;

            double smoothed = 0.5 * (grid[upper] + grid[lower]);
            double diff = fabs(grid[cell] - smoothed);

            my_max_diff = fmax(my_max_diff, diff);
            grid[cell] = smoothed;
        }
    }

    __shared__ double error_shared[256];  

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    error_shared[tid] = my_max_diff;
    __syncthreads();

    for (int step = blockDim.x * blockDim.y / 2; step > 0; step >>= 1) {
        if (tid < step) {
            error_shared[tid] = fmax(error_shared[tid], error_shared[tid + step]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        gpu_atomic_max(max_error_ptr, error_shared[0]); 
    }
}

void setup_boundaries(double* h_data, int dim_x, int dim_y, int dim_z) {
    for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
            for (int z = 0; z < dim_z; z++) {
                int idx = x*dim_y*dim_z + y*dim_z + z;
                if (x == 0 || x == dim_x-1 || y == 0 || y == dim_y-1 || z == 0 || z == dim_z-1) {
                    h_data[idx] = 10.0 * (x/(double)(dim_x-1) + 
                                        y/(double)(dim_y-1) + 
                                        z/(double)(dim_z-1));
                } else {
                    h_data[idx] = 0.0;
                }
            }
        }
    }
}

dim3 calculate_grid(dim3 block_size, int dim1, int dim2) {
    return dim3 (ceil(GRID_Y / (double)block_size.x), ceil(GRID_Z / (double)block_size.y));
}


typedef struct {
    double* data;
    int dim_x, dim_y, dim_z;
} Volume;

Volume create_volume(int x, int y, int z) {
    Volume vol;
    vol.dim_x = x;
    vol.dim_y = y;
    vol.dim_z = z;
    size_t size = x * y * z * sizeof(double);
    cudaCheck(cudaMalloc(&vol.data, size));
    return vol;
}

void free_volume(Volume vol) {
    cudaCheck(cudaFree(vol.data));
}

int main() {
    const dim3 def_block(16, 16);

    const dim3 x_grid = calculate_grid(def_block, GRID_Y, GRID_Z);
    const dim3 y_grid = calculate_grid(def_block, GRID_X, GRID_Z);
    const dim3 rearrange_grid = calculate_grid(def_block, GRID_X, GRID_Y);

    cudaDeviceProp props;
    cudaCheck(cudaGetDeviceProperties(&props, 0));
    printf("Compute Device: %s\n", props.name);
    printf("Total GPU Memory: %.2f GB\n", props.totalGlobalMem / (1024.0*1024.0*1024.0));

    Volume main_vol = create_volume(GRID_X, GRID_Y, GRID_Z);
    Volume temp_vol = create_volume(GRID_Y, GRID_Z, GRID_X); 
    
    double* h_data = (double*)malloc(GRID_X * GRID_Y * GRID_Z * sizeof(double));
    setup_boundaries(h_data, GRID_X, GRID_Y, GRID_Z);
    cudaCheck(cudaMemcpy(main_vol.data, h_data, 
                             GRID_X*GRID_Y*GRID_Z*sizeof(double), 
                             cudaMemcpyHostToDevice));

    double* d_error;
    cudaCheck(cudaMalloc(&d_error, sizeof(double)));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start));

    for (int iter = 1; iter <= MAX_ITERATIONS; iter++) {
        double error = 0.0;
        cudaCheck(cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice));

        process_x_axis<<<x_grid, def_block>>>(main_vol.data, main_vol.dim_x, main_vol.dim_y, main_vol.dim_z);
        process_y_axis<<<y_grid, def_block>>>(main_vol.data, main_vol.dim_x, main_vol.dim_y, main_vol.dim_z);
        rearrange_data<<<rearrange_grid, def_block>>>(main_vol.data, temp_vol.data, main_vol.dim_x, main_vol.dim_y, main_vol.dim_z);
        process_z_axis<<<rearrange_grid, def_block,def_block.x * def_block.y * sizeof(double)>>>(temp_vol.data,temp_vol.dim_x, temp_vol.dim_y, temp_vol.dim_z, d_error);
        rearrange_data<<<rearrange_grid, def_block>>>(temp_vol.data, main_vol.data, 
                                                          temp_vol.dim_z, temp_vol.dim_x, temp_vol.dim_y);

        cudaCheck(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost));
        printf("Iteration %4d, Error = " "%14.7E" "\n", iter, error);
        if (error < CONVERGENCE_THRESHOLD) break;
    }

    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float exec_time = 0.0f;
    cudaCheck(cudaEventElapsedTime(&exec_time, start, stop));

    printf("\nComputation Complete\n");
    printf("Grid Dimensions: %d x %d x %d\n", GRID_X, GRID_Y, GRID_Z);
    printf("Total Iterations: %d\n", MAX_ITERATIONS);
    printf("Time in seconds =  %.3f seconds\n", exec_time / 1000.0f);
    printf("Memory Usage: %.2f MB\n", (2.0 * GRID_X * GRID_Y * GRID_Z * sizeof(double)) / (1024.0*1024.0));

    free(h_data);
    free_volume(main_vol);
    free_volume(temp_vol);
    cudaCheck(cudaFree(d_error));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    return 0;
}