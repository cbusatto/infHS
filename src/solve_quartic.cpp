//////////////////////
// solve-quartic.cc //
//////////////////////

#include <RcppEigen.h>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#define pi 3.141592653589793

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;

// [[Rcpp::export]]

double solve_quartic (const Eigen::VectorXd& coefficients)
{
  std::complex<double> roots[4];
  
  const double a = coefficients(4);
  const double b = coefficients(3) / a;
  const double c = coefficients(2) / a;
  const double e = coefficients(0) / a;
  
  const double b2 = b * b;
  const double c2 = c * c;
  
  const std::complex<double> Q1 = c2 + 12.0 * e;
  const std::complex<double> Q2 = 2.0 * c2 * c + 27.0 * b2 * e - 72.0 * c * e;
  const std::complex<double> Q3 = 2.0 * b * (4.0 * c - b2);
  const std::complex<double> Q4 = 3.0 * b2 - 8.0 * c;
  
  const std::complex<double> Q5 = std::pow(Q2 / 2.0 + std::pow(Q2 * Q2 / 4.0 - Q1 * Q1 * Q1, 0.5), 1.0 / 3.0);
  const std::complex<double> Q6 = (Q1 / Q5 + Q5) / 3.0;
  const std::complex<double> Q7 = 2.0 * std::pow(Q4 / 12.0 + Q6, 0.5);
  
  const std::complex<double> t1 = 4.0 * Q4 / 6.0 - 4.0 * Q6;
  const std::complex<double> t2 = Q3 / Q7;
  
  const std::complex<double> ts1 = std::pow(t1 - t2, 0.5);
  const std::complex<double> ts2 = std::pow(t1 + t2, 0.5);
  
  roots[0] = (-b - Q7 - ts1) / 4.0;
  roots[1] = (-b - Q7 + ts1) / 4.0;
  roots[2] = (-b + Q7 - ts2) / 4.0;
  roots[3] = (-b + Q7 + ts2) / 4.0;
  
  int i = 0;
  double out = -1.0;
  while (out <= 0.0) {
    //Rcpp::Rcout << roots[i];
    if ((roots[i].real() > 0.0) && (std::abs(roots[i].imag()) < 1e-10)) out = roots[i].real();
    
    i += 1;
  }
  
  return out;
}


// [[Rcpp::export]]

double solve_quartic2 (const double& a1, const double& a2, const double& a3)
{
  // The algorithm below was derived by solving the quartic in Mathematica, and simplifying the resulting expression by hand.
  std::complex<double> roots[4];
  
  const double a = a3;
  const double b = a2 / a;
  const double c = -1.0 / a;
  const double e = a1 / a;
  
  const double b2 = b * b;
  const double c2 = c * c;
  
  const std::complex<double> Q1 = c2 + 12.0 * e;
  const std::complex<double> Q2 = 2.0 * c2 * c + 27.0 * b2 * e - 72.0 * c * e;
  const std::complex<double> Q3 = 2.0 * b * (4.0 * c - b2);
  const std::complex<double> Q4 = 3.0 * b2 - 8.0 * c;
  
  const std::complex<double> Q5 = std::pow(Q2 / 2.0 + std::pow(Q2 * Q2 / 4.0 - Q1 * Q1 * Q1, 0.5), 1.0 / 3.0);
  const std::complex<double> Q6 = (Q1 / Q5 + Q5) / 3.0;
  const std::complex<double> Q7 = 2.0 * std::pow(Q4 / 12.0 + Q6, 0.5);
  
  const std::complex<double> t1 = 4.0 * Q4 / 6.0 - 4.0 * Q6;
  const std::complex<double> t2 = Q3 / Q7;
  
  const std::complex<double> ts1 = std::pow(t1 - t2, 0.5);
  const std::complex<double> ts2 = std::pow(t1 + t2, 0.5);
  
  roots[0] = (-b - Q7 - ts1) / 4.0;
  roots[1] = (-b - Q7 + ts1) / 4.0;
  roots[2] = (-b + Q7 - ts2) / 4.0;
  roots[3] = (-b + Q7 + ts2) / 4.0;
  
  int i = 0;
  double out = -1.0;
  while (out <= 0.0) {
    //Rcpp::Rcout << roots[i];
    if ((roots[i].real() > 0.0) && (std::abs(roots[i].imag()) < 1e-10)) out = roots[i].real();
    
    i += 1;
  }
  
  return out;
}