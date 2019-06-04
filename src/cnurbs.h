/*--------------------------------------------------------------------------------------
 * A Digital Image Correlation library in Python and C that uses a global FE approach
 *
 * Sam Potter
 *
 *--------------------------------------------------------------------------------------
 *  nurbs.h: header file for functions defined in nurbs.c
 *-------------------------------------------------------------------------------------*/

/* Includes */

#include <string.h>
#include <math.h>

/* Function prototypes */

#ifndef CNURBS_H
#define CNURBS_H

int find_spanC(int num_ctrlpts, int degree, double knot, double knot_vector[]);
int basis_functionsC(double* N, int knot_span, double knot, int degree, double knot_vector[]);
double surface_pointC(int ncpts_u, int deg_u, double kv_u[], int ncpts_v, int deg_v, double kv_v[], double P[], double u, double v);

#endif
