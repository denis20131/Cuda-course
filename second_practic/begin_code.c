/* ADI program */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> // для измерения времени

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define nx 900
#define ny 900
#define nz 900

void init(double (*a)[ny][nz]);

int main(int argc, char *argv[])
{
    double maxeps, eps;
    double (*a)[ny][nz];
    int it, itmax, i, j, k;
    clock_t startt, endt;

    maxeps = 0.01;
    itmax = 10;
    a = (double (*)[ny][nz])malloc(nx * ny * nz * sizeof(double));
    init(a);

    startt = clock(); // начало измерения времени

    for (it = 1; it <= itmax; it++)
    {
        eps = 0;        
        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                    a[i][j][k] = (a[i-1][j][k] + a[i+1][j][k]) / 2;

        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                    a[i][j][k] = (a[i][j-1][k] + a[i][j+1][k]) / 2;

        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                {
                    double tmp1 = (a[i][j][k-1] + a[i][j][k+1]) / 2;
                    double tmp2 = fabs(a[i][j][k] - tmp1);
                    eps = Max(eps, tmp2);
                    a[i][j][k] = tmp1;
                }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }

    endt = clock(); // окончание измерения времени

    free(a);

    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2lf\n", (double)(endt - startt) / CLOCKS_PER_SEC);
    printf(" Operation type  =   double precision\n");

    printf(" END OF ADI Benchmark\n");
    return 0;
}

void init(double (*a)[ny][nz])
{
    int i, j, k;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++)
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[i][j][k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[i][j][k] = 0;
}
