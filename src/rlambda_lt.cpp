#include <RcppEigen.h>
#include <Eigen/Core>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]

#include "utils_rgamma3p.h"
#include "solve_quartic.h"


// [[Rcpp::export]]

double rlambda_lt (const double& d, const double& a, const double& b, const double& lt, const double& lambda_last, const int& eps = 1e+05)
{
  // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // sample values from distribution f(x) = c_f * x^(-1) * exp(-d/x^2 - a^2*x^2 + b*x) * I(x >= lt)
  
  const double a22 = 2.0 * a * a;
  const double d2 = 2 * d;
  
  int i = 0, c, cs;
  // int bl = 1;
  double xs, cs2, lksup, ys, u, lr;
  
  if (std::abs(b) < 1e-03) 
  {
    xs = std::sqrt(std::sqrt(1.0 + 8.0 * a22 * d) - 1.0) / (2.0 * a);
    c = std::round(xs * (a22 * xs - b));
    cs = c + 1;
    cs2 = 0.5 * (double)cs;
    
    // maximum of ratio r = f/g
    lksup = -cs2 * std::log(d2 / cs) - cs2;
    
    // :::::::::::::::::::::::::::::::::::
    // acceptance-rejection step
    // :::::::::::::::::::::::::::::::::::
    
    // while ((bl == 1) && (i < 10000))
    while (i < eps)
    {
      ys = std::sqrt(R::rgamma(cs2, 1.0 / (a * a)));
      
      if (ys >= lt)
      {
        u = R::runif(0.0, 1.0);
        lr = -cs * std::log(ys) - d / (ys * ys);
        if ((std::log(u) + lksup) <= lr) 
          return ys;
      }
      
      i += 1;
    }
  }
  
  // select parameter c
  xs = solve_quartic2(2.0 * d, b, -a22);
  c = std::round(xs * (a22 * xs - b));
  c = std::max(0, c);
  
  if (c < 1)
  {
    // maximum of ratio r = f/g
    lksup = -0.5 * std::log(d2) - 0.5;
    
    // :::::::::::::::::::::::::::::::::::
    // acceptance-rejection step
    // :::::::::::::::::::::::::::::::::::
    
    // int imax = 0;
    while (i < eps)
      // while (bl == 1)
    {
      ys = R::rnorm(b / a22, 1.0 / std::sqrt(a22));
      
      if (ys >= lt)
      {
        u = R::runif(0.0, 1.0);
        lr = -std::log(ys) - d / (ys * ys);
        
        if ((std::log(u) + lksup) <= lr)
          return ys;
      }
      
      i += 1;
    }
    
    lksup = -std::log(0.5 * d2) - 1.0;
    
    // while (bl == 1)
    while (i < eps)
    {
      ys = rg3p_c1(a, b);
      
      if (ys >= lt) 
      {
        u = R::runif(0.0, 1.0);
        lr = -2.0 * std::log(ys) - d / (ys * ys);
        
        if ((log(u) + lksup) <= lr) 
          return ys;
      }
     
      i += 1;
    }
  }
  
  cs = c + 1;
  cs2 = 0.5 * (double)cs;
  
  // maximum of ratio r = f/g
  lksup = -cs2 * std::log(d2 / cs) - cs2;
  
  // :::::::::::::::::::::::::::::::::::
  // acceptance-rejection step
  // :::::::::::::::::::::::::::::::::::
  
  if (c < 2)
  {
    while (i < eps)
      // while (bl == 1)
    {
      ys = rg3p_c1(a, b);
      
      if (ys >= lt) 
      {
        u = R::runif(0.0, 1.0);
        lr = -cs * std::log(ys) - d / (ys * ys);
        
        if ((log(u) + lksup) <= lr) 
          return ys;
      }
      
      i += 1;
    }
  }
  else
  {
    while (i < eps)
      // while (bl == 1)
    {
      ys = rg3p(a, b, c);
      
      if (ys >= lt) 
      {
        u = R::runif(0.0, 1.0);
        lr = -cs * std::log(ys) - d / (ys * ys);
        
        if ((log(u) + lksup) <= lr) 
          return ys;
      }
      
      i += 1;
    }
  }
  
  Rcpp::Rcout << "\n Warning: lambda set to last value \n";
  Rcpp::Rcout << "c: " << c;
  
  return lambda_last;
}
