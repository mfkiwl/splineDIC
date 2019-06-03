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

/* Function prototypes */

 #ifndef NURBS_H
 #define NURBS_H

 int find_span(int num_ctrlpts, int degree, double knot, double knot_vector[]);
 int basis_functions(double* N, int knot_span, double knot, int degree, double knot_vector[]);

 #endif
