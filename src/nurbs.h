/*--------------------------------------------------------------------------------------
 * A Digital Image Correlation library in Python and C that uses a global FE approach
 *
 * Sam Potter
 *
 *--------------------------------------------------------------------------------------
 *  nurbs.h: header file for functions defined in nurbs.c
 *-------------------------------------------------------------------------------------*/

/* Function prototypes */

 int find_span(int num_ctrlpts, int degree, double knot, int knot_vector);
 double basis_functions(int knot_span, double knot, int degree, double knot_vector);
