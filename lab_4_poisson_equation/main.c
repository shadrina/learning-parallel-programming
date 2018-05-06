#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>
#include "input.h"

// (I + 1) % P == 0 !
const int P = 8;

double init_area(double *f) {
    int index;
    for (int i = 0; i <= I; i++)
        for (int j = 0; j <= J; j++)
            for (int k = 0; k <= K; k++) {
                index = i * (J + 1) * (K + 1) + j * (K + 1) + k;
                if (i % I == 0 || j % J == 0 || k % K == 0) {
                    f[index] = boundary_func(i, j, k);
                } else {
                    f[index] = 0;
                }
            }
}

int main(int argc, char *argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0 && size != P) {
        fprintf(stderr, "Please, make sure you use %d processes!\n", P);
        exit(1);
    }

    // The number of "slices" per process
    int slice_width = (I + 1) / P;
    // The number of values in one "slice"
    int slice_area = (J + 1) * (K + 1);

    double *f;
    double *fbuff[2];
    fbuff[0] = (double *)malloc(slice_width * slice_area * sizeof(double));
    fbuff[1] = (double *)malloc(slice_width * slice_area * sizeof(double));
    count_denominator();

    if (rank == 0) {
        f = (double *)malloc((I + 1) * (J + 1) * (K + 1) * sizeof(double));
        init_area(f);
    }

    MPI_Scatter(f, slice_width * slice_area, MPI_DOUBLE, fbuff[0], slice_width * slice_area, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank >= 0) {
        MPI_Request up_request, down_request;
        MPI_Status up_status, down_status;
        for (int i = 0; i < slice_width * slice_area; i++) {
            fbuff[1][i] = fbuff[0][i];
        }

        double *up_neighbor_slice = (double *)malloc(slice_area * sizeof(double));
        double *down_neighbor_slice = (double *)malloc(slice_area * sizeof(double));

        struct timeval start, end;
        double fi, fj, fk;
        int prev = 0;
        int next = 1;
        int there_is_inaccurate_value = 1;
        int there_is_inaccurate_value_in_other_process = 1;

        gettimeofday(&start, NULL);
        while (there_is_inaccurate_value || there_is_inaccurate_value_in_other_process) {
            there_is_inaccurate_value = 0;

            if (rank != 0)        MPI_Isend(fbuff[next], slice_area, MPI_DOUBLE, rank - 1, 345, MPI_COMM_WORLD, &down_request);
            if (rank != size - 1) MPI_Isend(fbuff[next] + (slice_width - 1) * slice_area, slice_area, MPI_DOUBLE, rank + 1, 567, MPI_COMM_WORLD, &up_request);
            if (rank != 0)        MPI_Irecv(up_neighbor_slice, slice_area, MPI_DOUBLE, rank - 1, 567, MPI_COMM_WORLD, &up_request);
            if (rank != size - 1) MPI_Irecv(down_neighbor_slice, slice_area, MPI_DOUBLE, rank + 1, 345, MPI_COMM_WORLD, &down_request);

            for (int i = 0; i < (I + 1) / P; i++)
                for (int j = 1; j < J; j++)
                    for (int k = 1; k < K; k++) {

                        if (i == 0 || i == ((I + 1) / P) - 1) {
                            if (i == 0 && rank == 0 || (i == ((I + 1) / P) - 1 && rank == size - 1)) {
                                continue;
                            }
                            if (i == 0 && MPI_Wait(&up_request, &up_status) == MPI_SUCCESS) {
                                fi = (fbuff[prev][(i + 1) * (J + 1) * (K + 1) + j * (K + 1) + k]
                                    + up_neighbor_slice[j * (K + 1) + k]) / Hx2;
                            }
                            if (i == (((I + 1) / P) - 1) && MPI_Wait(&down_request, &down_status) == MPI_SUCCESS) {
                                fi = (down_neighbor_slice[j * (K + 1) + k]
                                    + fbuff[prev][(i - 1) * (J + 1) * (K + 1) + j * (K + 1) + k]) / Hx2;
                            } 
                        } else {
                            fi = (fbuff[prev][(i + 1) * (J + 1) * (K + 1) + j * (K + 1) + k]
                                + fbuff[prev][(i - 1) * (J + 1) * (K + 1) + j * (K + 1) + k]) / Hx2;
                        }

                        fj = (fbuff[prev][i * (J + 1) * (K + 1) + (j + 1) * (K + 1) + k]
                            + fbuff[prev][i * (J + 1) * (K + 1) + (j - 1) * (K + 1) + k]) / Hy2;
                        fk = (fbuff[prev][i * (J + 1) * (K + 1) + j * (K + 1) + (k + 1)]
                            + fbuff[prev][i * (J + 1) * (K + 1) + j * (K + 1) + (k - 1)]) / Hz2;

                        int index = i * (J + 1) * (K + 1) + j * (K + 1) + k; 
                        fbuff[next][index] = (fi + fj + fk - ro_func(i + rank * (I + 1) / P, j, k)) / denominator;

                        double error = fabs(fbuff[next][index] - fbuff[prev][index]);
                        if (error > e) {
                            there_is_inaccurate_value = 1;
                        }
                    }
                    prev = 1 - prev;
                    next = 1 - next;

            MPI_Allreduce(&there_is_inaccurate_value, &there_is_inaccurate_value_in_other_process, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }
        gettimeofday(&end, NULL);
        if (rank == 0) printf("Time taken: %lf\n", end.tv_sec - start.tv_sec + 0.0000001 * (end.tv_usec - start.tv_usec));

        double max_error = 0;
        for (int i = 0; i < (I + 1) / P; i++)
            for (int j = 1; j < J; j++)
                for (int k = 1; k < K; k++) {
                    int index = i * slice_area + j * (K + 1) + k;
                    if (fabs(fbuff[next][index] - boundary_func(i + rank * (I + 1) / P, j, k)) > max_error) {
                        max_error = fabs(fbuff[next][index] - 1.0 * boundary_func(i + rank * (I + 1) / P, j, k));
                    }
                }
        double common_max_error = 0;
        MPI_Allreduce(&max_error, &common_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("Max error: %f\n", common_max_error);

    }

    MPI_Finalize();

    return 0;
}
