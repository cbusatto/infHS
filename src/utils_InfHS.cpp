#include <RcppEigen.h>
#include <Eigen/Core>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]

#include "utils_rgamma3p.h"
#include "solve_quartic.h"


Eigen::VectorXd repelem (const double& x, const int& n) 
{
  Eigen::VectorXd out(n);
  for (int i = 0; i < n; i++) 
    out(i) = x;
  
  return out;
}

Eigen::VectorXd rmvnorm (const Eigen::VectorXd& m, const Eigen::MatrixXd& XtX, const Eigen::VectorXd& d, const double& sigma2, const int& n) 
{
  // generate multivariate normal sample N(Dev * m, Dev), Dev = (XtX + diag(d))^(-1)
  
  Eigen::MatrixXd Dev = XtX;
  Dev.diagonal() += d;
  
  const Eigen::MatrixXd R = Dev.llt().matrixL();
  
  Eigen::VectorXd z(n);
  for (int i = 0; i < n; i++) 
    z(i) = R::rnorm(0, 1);
  
  Eigen::VectorXd tmp = R.triangularView<Eigen::Lower>().solve(m);
  return R.transpose().triangularView<Eigen::Upper>().solve(std::sqrt(sigma2) * z + tmp);
}


// [[Rcpp::export]]

Eigen::VectorXd rmvnorm_b (const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& d, const double& sigma2) 
{
  // sample multivariate normal distribution x ~ N_p (S * t(X) * y, S), S = (t(X) * X + D^(-1))^(-1)

  const int n = X.rows(), p = X.cols();
  const double s = std::sqrt(sigma2);
  
  Eigen::VectorXd u(p);
  for (int i = 0; i < p; i++)
    u(i) = R::rnorm(0, std::sqrt(d(i)));

  Eigen::VectorXd v = X * u;

  const Eigen::MatrixXd dX = d.asDiagonal() * X.transpose();
  Eigen::MatrixXd S2 = X * dX;
  
  for (int i = 0; i < n; i++)
  {
    v(i) += R::rnorm(0, 1);    
    S2(i, i) += 1.0;
  }

  const Eigen::VectorXd w = S2.llt().solve(y / s - v);

  return s * (u + dX * w);
}


// [[Rcpp::export]]

double rlambda (const double& d, const double& a, const double& b, const double& lambda_last)
{
  // Rcpp::Rcout << "\n par:";
  // Rcpp::Rcout << "\n d: " << d;
  // Rcpp::Rcout << "\n a: " << a;
  // Rcpp::Rcout << "\n b: " << b;
  
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
    while (i < 1e+16)
    {
      ys = std::sqrt(R::rgamma(cs2, 1.0 / (a * a)));
      u = R::runif(0.0, 1.0);
      lr = -cs * std::log(ys) - d / (ys * ys);
      if ((std::log(u) + lksup) <= lr) 
        return ys;
      
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
    while (i < 1e+06)
      // while (bl == 1)
    {
      ys = R::rnorm(b / a22, 1.0 / std::sqrt(a22));
      
      if (ys > 0.0)
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
    while (i < 1e+06)
    {
      ys = rg3p_c1(a, b);
      u = R::runif(0.0, 1.0);
      lr = -2.0 * std::log(ys) - d / (ys * ys);
      
      if ((log(u) + lksup) <= lr)
        return ys;
      
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
    while (i < 1e+06)
    // while (bl == 1)
    {
      ys = rg3p_c1(a, b);
      u = R::runif(0.0, 1.0);
      lr = -cs * std::log(ys) - d / (ys * ys);
      
      if ((log(u) + lksup) <= lr) 
        return ys;
      
      i += 1;
    }
  }
  else
  {
    while (i < 1e+06)
    // while (bl == 1)
    {
      ys = rg3p(a, b, c);
      u = R::runif(0.0, 1.0);
      lr = -cs * std::log(ys) - d / (ys * ys);
      
      if ((log(u) + lksup) <= lr) 
        return ys;
      
      i += 1;
    }
  }

  // Rcpp::Rcout << "\n Warning: lambda set to last value \n";
  // Rcpp::Rcout << "c: " << c;
  
  return lambda_last;
}
