#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

const long long N = 4;
const double t = 0.01;
const double e = 0.0001;

// multithreading
const int P = 4;

// print
void print_vector(double *v, int length) {
    for (int i = 0; i < length; i++)
        printf("%f ", v[i]);
    printf("\n\n");
}
void print_matrix(double *M, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++)
            printf("%f ", M[i * N + j]);
        printf("\n");
    }
}

// initialize
void init_unit_matrix(double *M) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i == j) M[i * N + j] = 1;
            else M[i * N + j] = 0;
}
void init_part_of_unit_matrix(double *M, int rank) {
    for (int i = 0; i < (N / P); i++)
        for (int j = 0; j < N; j++)
            if (j == i + rank * N / P) M[i * N + j] = 1;
            else M[i * N + j] = 0;
}
void init_b_vector(double *b, int length) {
    for (int i = 0; i < length; i++)
        b[i] = i;
}
void nullify_vector(double *x, int length) {
    for (int i = 0; i < length; i++)
        x[i] = 0;
}

// multiply
void mult(double const *A, double const *b, double *result) {
    for (int i = 0; i < N; i++) result[i] = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            result[i] += A[i * N + j] * b[j];
}
void mult_part(double const *A, double const *b, double *result) {
    for (int i = 0; i < (N / P); i++) result[i] = 0;
    for (int i = 0; i < (N / P); i++)
        for (int j = 0; j < N; j++)
            result[i] += A[i * N + j] * b[j];
}

// count norm
double norm(double const *v, int length) {
    double result = 0;
    for (int i = 0; i < length; i++) {
        result += v[i] * v[i];
    }
    return sqrt(result);
}

// check solution
int check(double const *A, double const *b, double *result) {
    double Ax[N];
    mult(A, result, Ax);
    for (int i = 0; i < N; i++)
        Ax[i] = Ax[i] - b[i];
    if (norm(Ax, N) / norm(b, N) < e) return 1;
    return 0;
}
int check_part(double const *A, double const *b, double *result, int rank) {
    double Ax[N / P];
    mult_part(A, result, Ax);
    for (int i = 0; i < (N / P); i++)
        Ax[i] -= b[i + rank * (N / P)];
    double error = norm(Ax, (N / P)) / norm(b, N);
    if (error < e) return 1;
    return 0;
}

// approximate solution
void approximate(double const *A, double const *b, double const *prev, double *next) {
    mult(A, prev, next);
    for (int i = 0; i < N; i++)
        next[i] = next[i] - b[i];
    for (int i = 0; i < N; i++)
        next[i] = prev[i] - t * next[i];
}
void approximate_part(double const *A, double const *b, double const *prev, double *next, int rank) {
    mult_part(A, prev, next + rank * (N / P));
    for (int i = 0; i < (N / P); i++)
        next[i + rank * (N / P)] -= b[i + rank * (N / P)];
    for (int i = 0; i < (N / P); i++)
        next[i + rank * (N / P)] = prev[i + rank * (N / P)] - t * next[i + rank * (N / P)];
}

int main(int argc, char *argv[]) {
    // MPI
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // thread count
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // number of threads

    if (size != P) return 0;
    if (rank >= 0) {
        double *A = (double*)malloc((N / P) * N * sizeof(double));
        double *b = (double*)malloc(N * sizeof(double));
        double *prev = (double*)malloc(N * sizeof(double)); // previous x
        double *next = (double*)malloc(N * sizeof(double)); // next x

        init_part_of_unit_matrix(A, rank);
        init_b_vector(b, N);
        nullify_vector(prev, N);

        while (!check_part(A, b, next, rank)) {
            approximate_part(A, b, prev, next, rank);
            for (int i = rank * (N / P); i < (rank + 1) * (N / P); i++)
                prev[i] = next[i];
        }

        // At this stage, each thread has a vector with a portion of the calculated
        // coordinates of the required vector and zeros at the remaining positions.
        // All these vectors need to be sent to the service thread, which will add
        // them and calculate the final result.

        free(A);
        free(b);
        free(prev);
        free(next);
    }

    MPI_Finalize();
    return 0;
}
