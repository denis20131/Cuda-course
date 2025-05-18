/* Optimized ADI program with OpenMP parallelization */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define nx 900
#define ny 900
#define nz 900

void init(double (*a)[ny][nz]);

int main(int argc, char *argv[])
{
    double maxeps, eps;
    double (*a)[ny][nz];
    int it, itmax;

    maxeps = 0.01;
    itmax = 10;
    a = (double (*)[ny][nz])malloc(nx * ny * nz * sizeof(double));
    init(a);

    double startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++)
    {
        eps = 0;

        // Phase 1: X-direction
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++)
                for (int i = 1; i < nx - 1; i++)
                    a[i][j][k] = (a[i - 1][j][k] + a[i + 1][j][k]) * 0.5;

        // Phase 2: Y-direction
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < nx - 1; i++)
            for (int k = 1; k < nz - 1; k++)
                for (int j = 1; j < ny - 1; j++)
                    a[i][j][k] = (a[i][j - 1][k] + a[i][j + 1][k]) * 0.5;

        // Phase 3: Z-direction + EPS calculation
        #pragma omp parallel for collapse(2) reduction(max:eps) schedule(static)
        for (int i = 1; i < nx - 1; i++)
            for (int j = 1; j < ny - 1; j++)
            {
                double local_eps = 0.0;
                for (int k = 1; k < nz - 1; k++)
                {
                    double tmp1 = (a[i][j][k - 1] + a[i][j][k + 1]) * 0.5;
                    double tmp2 = fabs(a[i][j][k] - tmp1);
                    local_eps = Max(local_eps, tmp2);
                    a[i][j][k] = tmp1;
                }
                if (local_eps > eps) eps = local_eps;
            }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }

    double endt = omp_get_wtime();

    free(a);

    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2lf\n", (double)(endt - startt));
    printf(" Operation type  =   double precision\n");

    printf(" END OF ADI Benchmark\n");
    return 0;
}

void init(double (*a)[ny][nz])
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[i][j][k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[i][j][k] = 0;
}