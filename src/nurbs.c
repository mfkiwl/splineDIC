/*--------------------------------------------------------------------------------------
 * A Digital Image Correlation library in Python and C that uses a global FE approach
 *
 * Sam Potter
 *
 *--------------------------------------------------------------------------------------
 *  nurbs.c: C module for basic B-spline surface constructs
 *-------------------------------------------------------------------------------------*/
 
 #include "nurbs.h"
 #include <stdlib.h>

 /*-------------------------------------------------------------------------------------
  * find_span: function for determining the knot span given a knot value.
  *-------------------------------------------------------------------------------------*/

 int find_span(int num_ctrlpts, int degree, double knot, int knot_vector)
 {
    double rtol = 1E-6;
    if(abs(knot - knot_vector[num_ctrlpts]) <= rtol){
        return num_ctrlpts - 1;
    }

    // Begin binary search
    int low = degree;
    int high = num_ctrlpts;
    
    // Compute midpoint sum
    int mid_sum = low + high;

    // Case structure on whether mid_sum is odd or even
    double mid;
    if(mid_sum % 2 == 0){
        mid = mid_sum * 0.5;
    } else {
        mid = (mid_sum + 1) * 0.5;
    }

    // Cast result to int so it works as an index
    int mid_index = (int) mid;

    // While loop to perform binary search
    while(knot < knot_vector[mid_index] || knot > knot_vector[mid_index + 1]){
        // Update high/low
        if(knot < knot_vector[mid_index]){
            high = mid_index;
        } else {
            low = mid_index;
        }

        // Update mid value
        mid_index = (int) ((low + high) * 0.5);
    }
    
    return mid_index;
}
        
/*-------------------------------------------------------------------------------------
  * basis_functions: function for determining the value of a non-zero basis functions at a given knot value
  *-------------------------------------------------------------------------------------*/

  double basis_functions(int knot_span, double knot, int degree, double knot_vector)
 {
	 double N[degree + 1] = { 1 };
	 double left[degree + 1] = { 0 };
	 double right[degree + 1] = { 0 };
	 double saved;
	 double temp;
	 
	 int j;
	 for(j=1; j<=degree; j++){
		 left[j] = knot - knot_vector[knot_span + 1 - j];
		 right[j] knot_vector[knot_span + j] - knot;
		 saved = 0.0;
		 
		 for(r=0; r<=j; r++){
			 temp = N[r] / (right[r + 1] + left[j - r]);
			 N[r] = saved + right[r + 1] * temp;
			 saved = left[j - r] * temp;
		 }
		 
		 N[j] = saved;
		 
	 }
	 
	 return N;
 }

