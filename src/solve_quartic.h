
/////////////////////
// solve-quartic.h //
/////////////////////

#ifndef solve_quartic_h
#define solve_quartic_h

// The solve_quartic routine solves the generic quartic equation:
//
//     a * x^4 + b * x^3 + c * x^2 + d * x + e == 0
//
// Usage:
//
//     solve_quartic({e, 0, -1, b, a}, roots).

double solve_quartic (const Eigen::VectorXd& coefficients);
double solve_quartic2 (const double& a1, const double& a2, const double& a3);
  
#endif // solve_quartic_h
