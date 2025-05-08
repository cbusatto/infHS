// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //
// ::::::::::::::::::::                                              :::::::::::::::::::: //
// ::::::::::::::::::::    Reversible Jump: SS variable selection    :::::::::::::::::::: //
// ::::::::::::::::::::                                              :::::::::::::::::::: //
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

#include <RcppEigen.h>
#include <Eigen/Core>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;


// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// sample truncated normal variables

void normstdrnd(double *t1, double *t2) 
{
  double z, u2;
  
  u2 = unif_rand();
  if (!((u2 > 0.0) && (u2 < 1.0))) 
    u2 = unif_rand();

  z   = std::sqrt((-2.0 * (std::log(u2))));
  u2  = 2.0 * M_PI * unif_rand();
  *t1 = (double) (z * std::cos(u2));
  *t2 = (double) (z * std::sin(u2));
}

double ltnorm(double mean, double var) 
{
  double x, ran1, ran2;
  bool stop;
  double lb = -mean / std::sqrt(var);
  
  stop = false;
  while (stop == false) 
  {
    normstdrnd(&ran1, &ran2);
    x = ran1;
    stop = (x > lb);
  }
  
  x *= std::sqrt(var);
  x += mean;
  
  return(x);
}

double rtnorm(double mean, double var) 
{
  double x, ran1, ran2;
  bool stop;
  double ub = -mean / sqrt(var);
  
  stop = FALSE;
  while (stop == FALSE) 
  {
    normstdrnd(&ran1, &ran2);
    x = ran1;
    stop = (x < ub);
  }
  
  x *= std::sqrt(var);
  x += mean;
  
  return(x);
}

