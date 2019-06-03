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
int basis_functions(double* N, int knot_span, double knot, int degree, double knot_vector[]);

#endif
