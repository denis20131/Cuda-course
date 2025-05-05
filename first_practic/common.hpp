#ifndef COMMON_HPP
#define COMMON_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef USE_FLOAT
typedef float real;
#define EPSILON 1e-5f
#else
typedef double real;
#define EPSILON 1e-10
#endif

#define IDX(i, j, k, L) ((i) * (L) * (L) + (j) * (L) + (k))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#endif
