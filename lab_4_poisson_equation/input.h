#ifndef OPP_LAB_4_INPUT_H
#define OPP_LAB_4_INPUT_H

#include <math.h>

#define X_LENGTH 1
#define Y_LENGTH 1
#define Z_LENGTH 1

#define I 239
#define J 239
#define K 239

#define a 1000000
#define e 0.00000001

double Hx, Hy, Hz;
double Hx2, Hy2, Hz2;
double denominator;

double boundary_func(double i, double j, double k) {
    return i * i + j * j + k * k;
}


double ro_func(double i, double j, double k) {
    return (6 + a * boundary_func(i, j, k));
}

void initialize_spaces(void) {
    Hx = X_LENGTH * 1.0 / I;
    Hx2 = Hx * Hx;
    Hy = Y_LENGTH * 1.0 / J;
    Hy2 = Hy * Hy;
    Hz = Z_LENGTH * 1.0 / K;
    Hz2 = Hz * Hz;
}

void count_denominator(void) {
    initialize_spaces();
    denominator = 2.0 / Hx2 + 2.0 / Hy2 + 2.0 / Hz2 - a;
}

#endif //OPP_LAB_4_INPUT_H

