#include "common.hpp"
#include <omp.h>
#include <vector>
#include <cstdlib> // для atoi

int main(int argc, char** argv) {
    // Проверка аргументов командной строки
    if (argc < 3) {
        printf("Usage: %s L ITMAX\n", argv[0]);
        return 1;
    }

    // Инициализация параметров
    const int L = atoi(argv[1]);          // Размер сетки
    const int ITMAX = atoi(argv[2]);      // Макс. число итераций
    double MAXEPS = 0.5f;

    std::vector<real> A(L * L * L, 0), B(L * L * L, 0);
    real eps;

    // Инициализация
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                int idx = IDX(i, j, k, L);
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[idx] = 0;
                else
                    B[idx] = 4 + i + j + k;
            }

    double start = omp_get_wtime();

    for (int it = 0; it < ITMAX; ++it) {
        eps = 0.0;

        #pragma omp parallel for collapse(3) reduction(max : eps)
        for (int i = 1; i < L - 1; ++i)
            for (int j = 1; j < L - 1; ++j)
                for (int k = 1; k < L - 1; ++k) {
                    int idx = IDX(i, j, k, L);
                    real tmp = std::abs(B[idx] - A[idx]);
                    eps = MAX(tmp, eps);
                    A[idx] = B[idx];
                }

        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L - 1; ++i)
            for (int j = 1; j < L - 1; ++j)
                for (int k = 1; k < L - 1; ++k) {
                    int idx = IDX(i, j, k, L);
                    B[idx] = (A[IDX(i-1,j,k,L)] + A[IDX(i,j-1,k,L)] + A[IDX(i,j,k-1,L)] +
                              A[IDX(i,j,k+1,L)] + A[IDX(i,j+1,k,L)] + A[IDX(i+1,j,k,L)]) / 6.0;
                }

        printf(" IT = %4d   EPS = %14.7E\n", it+1, eps);
        
        if (eps < MAXEPS)
            break;
    }

    double end = omp_get_wtime();
    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      = %20d\n", ITMAX);
    printf(" Time in seconds = %20.2f\n", end - start);
    printf(" END OF Jacobi3D Benchmark\n");
    
    return 0;
}