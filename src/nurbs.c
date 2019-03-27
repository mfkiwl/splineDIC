/*--------------------------------------------------------------------------------------
 * A Digital Image Correlation library in Python and C that uses a global FE approach
 *
 * Sam Potter
 *
 *--------------------------------------------------------------------------------------
 *  nurbs.c: C module for basic B-spline surface constructs
 *-------------------------------------------------------------------------------------*/
 
 #include nurbs.h
 
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