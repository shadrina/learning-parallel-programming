#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int N1 = 2;
const int N2 = 3;
const int N3 = 4;

const int P1 = 1;
const int P2 = 4;

const int MOD = 5;

/**
 * The program realizes the parallel matrix multiplication
 *
 * A: size(N1 * N2), strips(P1) - first operand
 * B: size(N2 * N3), strips(P2) - second operand
 * C: size(N1 * N3), parts(P1 * P2) - result
 *
 * The process with rank 0 is a service process.
 * It supplies all other processes with the necessary
 * parts of the matrices A and B.
 *
 */

void print_matrix(double *M, int rows, int columns);
void init_matrix(double *M, int rows, int columns);
void copy_matrix(double *Src, double *Dst, int rows, int columns);
void transpose_matrix(double *M, int rows, int columns);
void multiply_matrices(double *A, int A_rows, int A_columns, double *B, int B_rows, int B_columns, double *C);
void place_matrix_part(double *M_part, int M_part_rows, int M_part_columns, double *M, int rows, int columns, int rank);

int main(int argc, char **argv) {
    // MPI
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // process count
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // number of processes

    if (size != P1 * P2 && rank == 0) {
        fprintf(stderr, "Please, make sure you use %d processes!\n", P1 * P2);
        exit(EXIT_FAILURE);
    }

    // The service rank initializes both matrices A and B
    if (rank == 0) {
        double *A = (double*)malloc(N1 * N2 * sizeof(double));
        double *B = (double*)malloc(N2 * N3 * sizeof(double));
        double *C = (double*)malloc(N1 * N3 * sizeof(double));
        init_matrix(A, N1, N2);
        init_matrix(B, N2, N3);

        printf("A: \n");
        print_matrix(A, N1, N2);
        printf("B: \n");
        print_matrix(B, N2, N3);

        // Before sending the B matrix strips we need to transpose it
        // (It is difficult to cut the matrix into vertical strips)
        transpose_matrix(B, N2, N3);

        // Sending...
        double *A_shifted, *B_shifted;
        for (int i_rank = 1; i_rank < size; i_rank++) {
            // (N1 / P1) * N2 - size of the A's strip
            A_shifted = A + (i_rank / P2) * ((N1 / P1) * N2);
            MPI_Send(A_shifted, (N1 / P1) * N2, MPI_DOUBLE, i_rank, 123, MPI_COMM_WORLD);

            // (N3 / P2) * N2 - size of the B's transposed strip
            B_shifted = B + (i_rank % P2) * ((N3 / P2) * N2);
            MPI_Send(B_shifted, (N3 / P2) * N2, MPI_DOUBLE, i_rank, 123, MPI_COMM_WORLD);
        }

        // The service process also performs the role of an ordinary process
        double *B_strip = (double*)malloc(N2 * (N3 / P2) * sizeof(double));
        double *C_part =  (double*)malloc((N1 / P1) * (N3 / P2) * sizeof(double));

        copy_matrix(B, B_strip, N2, N3 / P2);
        transpose_matrix(B_strip, N3 / P2, N2);
        multiply_matrices(A, N1 / P1, N2, B_strip, N2, N3 / P2, C_part);
        place_matrix_part(C_part, N1 / P1, N3 / P2, C, N1, N3, 0);

        // Receiving results from other processes and storing them in C matrix
        for (int i_rank = 1; i_rank < size; i_rank++) {
            MPI_Recv(C_part, (N1 / P1) * (N3 / P2), MPI_DOUBLE, i_rank, 321, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            place_matrix_part(C_part, N1 / P1, N3 / P2, C, N1, N3, i_rank);
        }

        printf("C:\n");
        print_matrix(C, N1, N3);

        free(A);
        free(B);
        free(B_strip);
        free(C_part);
        free(C);
    }
    if (rank > 0) {
        double *A_strip = (double*)malloc((N1 / P1) * N2 * sizeof(double));
        double *B_strip = (double*)malloc(N2 * (N3 / P2) * sizeof(double));
        double *C_part =  (double*)malloc((N1 / P1) * (N3 / P2) * sizeof(double));

        MPI_Recv(A_strip, (N1 / P1) * N2, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_strip, (N3 / P2) * N2, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // The B_strip came to us in the transposed version
        transpose_matrix(B_strip, N3 / P2, N2);

        // Calculating the result and sending it to the service process
        multiply_matrices(A_strip, N1 / P1, N2, B_strip, N2, N3 / P2, C_part);
        MPI_Send(C_part, (N1 / P1) * (N3 / P2), MPI_DOUBLE, 0, 321, MPI_COMM_WORLD);

        printf("I am process #%d\n", rank);
        print_matrix(C_part, N1 / P1, N3 / P2);

        free(A_strip);
        free(B_strip);
        free(C_part);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void print_matrix(double *M, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++)
            printf("%.1f ", M[i * columns + j]);
        printf("\n");
    }
}

void init_matrix(double *M, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0;j < columns; j++) {
            M[i * columns + j] = rand() % MOD + 1;
        }
    }
}

void copy_matrix(double *Src, double *Dst, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            Dst[i * columns + j] = Src[i * columns + j];
        }
    }
}

void transpose_matrix(double *M, int rows, int columns) {
    double *M_transposed = (double*)malloc(columns * rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            M_transposed[j * rows + i] = M[i * columns + j];
        }
    }
    copy_matrix(M_transposed, M, rows, columns);
    free(M_transposed);
}

void multiply_matrices(double *A, int A_rows, int A_columns, double *B, int B_rows, int B_columns, double *C) {
    if (A_columns != B_rows) {
        fprintf(stderr, "Impossible to multiply the matrices");
        exit(EXIT_FAILURE);
    }
    int element;
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_columns; j++) {
            element = 0;
            for (int k = 0; k < A_columns; k++) {
                element += A[i * A_columns + k] * B[k * B_columns + j];
            }
            C[i * B_columns + j] = element;
        }
    }
}

void place_matrix_part(double *M_part, int M_part_rows, int M_part_columns, double *M, int rows, int columns, int rank) {
    for (int i = 0; i < M_part_rows; i++) {
        for (int j = 0; j < M_part_columns; j++) {
            M[(i + (rank / P2)) * columns + (j + (rank % P2) * M_part_columns)] = M_part[i * M_part_columns + j];
        }
    }
}