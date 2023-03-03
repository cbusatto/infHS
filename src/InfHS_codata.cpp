#include <Rcpp.h>
#include <RcppEigen.h>
#include <math.h>     

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]

#include "utils_InfHS.h"
#include "utils_InfHS_old.h"
#include "utils_truncnorm.h"


// Rcpp::List infHS_codata (const int& B, const int& bn, const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Rcpp::List Z_list, const Eigen::VectorXd& hyp_sigma, const Eigen::VectorXd& a_k, const Eigen::VectorXd& b_k, const double& s0 = 1, const int& ping = 1000)
// {
//   // B: number of iterations
//   // bn: number of burn-in iterations
//   // y: numeric response variable, n-dim vector
//   // X: design matrix, n x p matrix
//   // Z: list of p x m_d co-data matrices, d = 1, ..., D, sum_d m_d = M
//   // hyp_sigma: hyperparameters for sigma2
//   // a_k, b_k: D-dimensional vector, hyperparameters for \kappa_d, d = 1, ..., D
//   // s0: optional prior scale parameter for local shrinkage parameters lambda, default equal to 1 (Horseshoe)
//   // ping: optional integer, print the number of iterations completed each "ping" iterations, default is ping = 1000
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // initialization
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   int b, j, d;
//   double tmp_double, bl2_sum;
// 
//   const int n = y.size(), p = X.cols(), D = a_k.size();
//     
//   const double tau2_shape = 0.5 * ((double)p + 1.0), v = hyp_sigma(0), q = hyp_sigma(1);
//   double tau2 = 1.0, eta = 1.0 / R::rgamma(0.5, 1.0), sigma2 = 1.0;
//   
//   Eigen::VectorXd beta(p), lambda(p), phi2(p);
//   Eigen::VectorXd beta2(p), lambda2(p), lambda2_inv(p);
//   lambda.setOnes();
//   
//   Eigen::VectorXi md(D), md_index(D+1), am(D);
//   
//   md_index(0) = 0;
//   for (d = 0; d < D; d++) 
//   {
//     Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
//     md(d) = Zd.cols();
//     md_index(d+1) = md_index(d) + md(d);
//     am(d) = a_k(d) + 0.5 * (double)md(d);
//   }
//   
//   const int M = md_index(D);
//   
//   Eigen::VectorXd gamma(M), kappa(D), ds(M);
//   Eigen::MatrixXd Z(p, M);
//   kappa.setOnes(); gamma.setZero();
// 
//   for (d = 0; d < D; d++) 
//   {
//     Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
//     Z.block(0, md_index(d), p, md(d)) = Zd;
//   }
//   
//   Eigen::VectorXd qs(n);
//   Eigen::MatrixXd Zs(M, p);
//   const Eigen::VectorXd Xty = X.transpose() * y;
//   const Eigen::MatrixXd XtX = X.transpose() * X;
//   
//   beta = rmvnorm(Xty, XtX, lambda, sigma2, p);
//   
//   for (j = 0; j < p; j++)
//   {
//     beta2(j) = std::pow(beta(j), 2);
//     phi2(j) = 1.0 / R::rgamma(0.5, 2.0);
//   }
//   
//   Eigen::VectorXd s0phi2_inv(p), s0phi2 = s0 * phi2;
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // output objects
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   Eigen::VectorXd tau2_out(B-bn), sigma2_out(B-bn);
//   Eigen::MatrixXd Beta_out(B-bn, p), Lambda_out(B-bn, p), Gamma_out(B-bn, M);
//     
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // main cycle
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// 
//   for (b = 0; b < B; b++)
//   {
//     // sample lambda and phi2
//     bl2_sum = 0.0;
// 
//     for (j = 0; j < p; j++)
//     {
//       tmp_double = (Z.row(j) * gamma).value();
// 
//       beta2(j) = std::pow(beta(j), 2);
//       
//       lambda(j) = rlambda_old(0.5 * beta2(j) / (sigma2 * tau2), 1.0 / std::sqrt(2.0 * s0phi2(j)), tmp_double / s0phi2(j));
//       lambda2(j) = std::pow(lambda(j), 2);
//       lambda2_inv(j) = 1.0 / lambda2(j);
// 
//       bl2_sum += beta2(j) * lambda2_inv(j);
// 
//       phi2(j) = 1.0 / R::rgamma(1.0, 1.0 / (0.5 * (1.0 + std::pow(lambda(j) - tmp_double, 2) / s0)));
//       s0phi2(j) = s0 * phi2(j);
//       s0phi2_inv(j) = 1.0 / s0phi2(j);
//     }
//     
//     // sample gamma
//     Zs = Z.transpose() * s0phi2_inv.asDiagonal();
//     for (d = 0; d < D; d++)
//       ds.segment(md_index(d), md(d)) = repelem(1.0 / kappa(d), md(d));
// 
//     gamma = rmvnorm(Zs * lambda, Zs * Z, ds, s0, M);
//     
//     // sample kappa
//     for (d = 0; d < D; d++)
//       kappa(d) = 1.0 / R::rgamma(am(d), 1.0 / (b_k(d) + 0.5 * gamma.segment(md_index(d), md(d)).squaredNorm()));
// 
//     // sample tau2
//     tau2 = 1.0 / R::rgamma(tau2_shape, 1.0 / (1.0 / eta + 0.5 * bl2_sum / sigma2));
//     
//     // sample eta
//     eta = 1.0 / R::rgamma(0.5, 1.0 / (1.0 + 1.0 / tau2));
// 
//     // sample sigma2
//     qs = y - X * beta;
//     sigma2 = 1.0 / R::rgamma(0.5 * (n + p) + v, 1.0 / (q + 0.5 * (qs.squaredNorm() + bl2_sum / tau2)));
//     
//     // sample beta
//     beta = rmvnorm(Xty, XtX, (1.0 / tau2) * lambda2_inv, sigma2, p);
//     
//     if (b >= bn)
//     {
//       Beta_out.row(b-bn) = beta.transpose();
//       Lambda_out.row(b-bn) = lambda.transpose();
//       Gamma_out.row(b-bn) = gamma.transpose();
//       tau2_out(b-bn) = tau2;
//       sigma2_out(b-bn) = sigma2;
//     }
// 
//     if ((ping != 0) && (b % ping == 0))
//       Rcpp::Rcout << "\n iter: " << b;
//   }
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // output
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   return Rcpp::List::create(Rcpp::Named("Beta") = Beta_out, Rcpp::Named("Lambda") = Lambda_out, Rcpp::Named("sigma2") = sigma2_out, Rcpp::Named("tau2") = tau2_out, Rcpp::Named("Gamma") = Gamma_out);
// }



// Rcpp::List infHS_codata_fast (const int& B, const int& bn, const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Rcpp::List Z_list, const int& M, const Eigen::VectorXd& hyp_sigma, const Eigen::VectorXd& a_k, const Eigen::VectorXd& b_k, const double& s0 = 1, const int& ping = 1000)
// {
//   // B: number of iterations
//   // bn: number of burn-in iterations
//   // y: numeric response variable, n-dim vector
//   // X: design matrix, n x p matrix
//   // Z: list of p x m_d co-data matrices, d = 1, ..., D, sum_d m_d = M
//   // M: number of total columns in Z_list
//   // hyp_sigma: hyperparameters for sigma2
//   // a_k, b_k: D-dimensional vector, hyperparameters for \kappa_d, d = 1, ..., D
//   // s0: optional prior scale parameter for local shrinkage parameters lambda, default equal to 1 (Horseshoe)
//   // ping: optional integer, print the number of iterations completed each "ping" iterations, default is ping = 1000
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // initialization
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   int b, j, d;
//   double tmp_double, bl2_sum;
//   
//   const int n = y.size(), p = X.cols(), D = a_k.size();
//   
//   const double tau2_shape = 0.5 * ((double)p + 1.0), v = hyp_sigma(0), q = hyp_sigma(1);
//   double tau2 = 1.0, eta = 1.0, sigma2 = 1.0;
//   
//   Eigen::VectorXd beta(p), lambda(p), phi2(p), gamma(M), kappa(D), ds(M);
//   Eigen::VectorXd beta2(p), lambda2(p), lambda2_inv(p);
//   lambda.setOnes(); kappa.setOnes(); gamma.setZero();
//   
//   Eigen::VectorXi md(D), md_index(D+1), am(D);
//    
//   Eigen::VectorXd qs(n);
//   Eigen::MatrixXd Z(p, M), Zs(M, p);
//   
//   md_index(0) = 0;
//   for (d = 0; d < D; d++) 
//   {
//     Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
//     
//     md(d) = Zd.cols();
//     md_index(d+1) = md_index(d) + md(d);
//     am(d) = a_k(d) + 0.5 * (double)md(d);
//     
//     Z.block(0, md_index(d), p, md(d)) = Zd;
//   }
//   
//   // beta = rmvnorm_b(X, y, lambda, sigma2);
//   for (j = 0; j < p; j++)
//   {
//     beta(j) = R::rnorm(1, 0);
//     beta2(j) = std::pow(beta(j), 2);
//     phi2(j) = 1.0;
//   }
//   
//   Eigen::VectorXd s0phi2_inv(p), s0phi2 = s0 * phi2;
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // output objects
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   Eigen::VectorXd tau2_out(B-bn), sigma2_out(B-bn);
//   Eigen::MatrixXd Beta_out(B-bn, p), Lambda_out(B-bn, p), Gamma_out(B-bn, M);
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // main cycle
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   for (b = 0; b < B; b++)
//   {
//     // sample lambda and phi2
//     bl2_sum = 0.0;
//     
//     for (j = 0; j < p; j++)
//     {
//       tmp_double = (Z.row(j) * gamma).value();
//       
//       beta2(j) = std::pow(beta(j), 2);
//       
//       lambda(j) = rlambda_old(0.5 * beta2(j) / (sigma2 * tau2), 1.0 / std::sqrt(2.0 * s0phi2(j)), tmp_double / s0phi2(j));
//       lambda2(j) = std::pow(lambda(j), 2);
//       lambda2_inv(j) = 1.0 / lambda2(j);
//       
//       bl2_sum += beta2(j) * lambda2_inv(j);
//       
//       phi2(j) = 1.0 / R::rgamma(1.0, 1.0 / (0.5 * (1.0 + std::pow(lambda(j) - tmp_double, 2) / s0)));
//       s0phi2(j) = s0 * phi2(j);
//       s0phi2_inv(j) = 1.0 / s0phi2(j);
//     }
//     
//     // sample gamma
//     Zs = Z.transpose() * s0phi2_inv.asDiagonal();
//     for (d = 0; d < D; d++)
//       ds.segment(md_index(d), md(d)) = repelem(1.0 / kappa(d), md(d));
//     
//     gamma = rmvnorm(Zs * lambda, Zs * Z, ds, s0, M);
//     
//     // sample kappa
//     for (d = 0; d < D; d++)
//       kappa(d) = 1.0 / R::rgamma(am(d), 1.0 / (b_k(d) + 0.5 * gamma.segment(md_index(d), md(d)).squaredNorm()));
//     
//     // sample tau2
//     tau2 = 1.0 / R::rgamma(tau2_shape, 1.0 / (1.0 / eta + 0.5 * bl2_sum / sigma2));
//     
//     // sample eta
//     eta = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / tau2));
//     
//     // sample sigma2
//     qs = y - X * beta;
//     sigma2 = 1.0 / R::rgamma(0.5 * (n + p) + v, 1.0 / (q + 0.5 * (qs.squaredNorm() + bl2_sum / tau2)));
//     
//     // sample beta
//     beta = rmvnorm_b(X, y, tau2 * lambda2, sigma2);
//     
//     if (b >= bn)
//     {
//       Beta_out.row(b-bn) = beta.transpose();
//       Lambda_out.row(b-bn) = lambda.transpose();
//       Gamma_out.row(b-bn) = gamma.transpose();
//       tau2_out(b-bn) = tau2;
//       sigma2_out(b-bn) = sigma2;
//     }
//     
//     if ((ping != 0) && (b % ping == 0))
//       Rcpp::Rcout << "\n iter: " << b;
//   }
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // output
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   return Rcpp::List::create(Rcpp::Named("Beta") = Beta_out, Rcpp::Named("Lambda") = Lambda_out, Rcpp::Named("sigma2") = sigma2_out, Rcpp::Named("tau2") = tau2_out, Rcpp::Named("Gamma") = Gamma_out);
// }


// [[Rcpp::export]]

Rcpp::List infHS_FB (const int& B, const int& bn, const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Rcpp::List Z_list, const int& M, const Eigen::VectorXd& hyp_sigma, const Eigen::VectorXd& a_k, const Eigen::VectorXd& b_k, const double& s0 = 1, const int& ping = 1000)
{
  // B: number of iterations
  // bn: number of burn-in iterations
  // y: numeric response variable, n-dim vector
  // X: design matrix, n x p matrix
  // Z: list of p x m_d co-data matrices, d = 1, ..., D, sum_d m_d = M
  // M: number of total columns in Z_list
  // hyp_sigma: hyperparameters for sigma2
  // a_k, b_k: D-dimensional vector, hyperparameters for \kappa_d, d = 1, ..., D
  // s0: optional prior scale parameter for local shrinkage parameters lambda, default equal to 1 (Horseshoe)
  // ping: optional integer, print the number of iterations completed each "ping" iterations, default is ping = 1000
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // initialization
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  int b, j, d;
  double tmp_double, bl2_sum;
  
  const int n = y.size(), p = X.cols(), D = a_k.size();
  
  const double tau2_shape = 0.5 * ((double)p + 1.0), v = hyp_sigma(0), q = hyp_sigma(1);
  double tau2 = 1.0, eta = 1.0, sigma2 = 1.0, psi0 = 1.0;
  
  Eigen::VectorXd beta(p), lambda(p), phi2(p-1), gamma(M), kappa(D), ds(M);
  Eigen::VectorXd beta2(p), lambda2(p), lambda2_inv(p);
  lambda.setOnes(); kappa.setOnes(); gamma.setZero();
  
  Eigen::VectorXi md(D), md_index(D+1), am(D);
  
  Eigen::VectorXd qs(n);
  Eigen::MatrixXd Z(p-1, M), Zs(M, p-1);
  
  md_index(0) = 0;
  for (d = 0; d < D; d++) 
  {
    Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
    
    md(d) = Zd.cols();
    md_index(d+1) = md_index(d) + md(d);
    am(d) = a_k(d) + 0.5 * (double)md(d);
    
    Z.block(0, md_index(d), p-1, md(d)) = Zd;
  }
  
  // beta = rmvnorm_b(X, y, lambda, sigma2);
  for (j = 0; j < p; j++)
  {
    beta(j) = R::rnorm(1, 0);
    beta2(j) = std::pow(beta(j), 2);
    if (j < p-1)
      phi2(j) = 1.0;
  }
  
  Eigen::VectorXd s0phi2_inv(p-1), s0phi2 = s0 * phi2;
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // output objects
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  Eigen::VectorXd tau2_out(B-bn), sigma2_out(B-bn);
  Eigen::MatrixXd Beta_out(B-bn, p), Lambda_out(B-bn, p), Gamma_out(B-bn, M);
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // main cycle
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  for (b = 0; b < B; b++)
  {
    // sample lambda and phi2
    bl2_sum = 0.0;
    
    lambda2(0) = 1.0 / R::rgamma(1.0, 1.0 / (1.0 / psi0 + 0.5 * (std::pow(beta(0), 2) / (sigma2 * tau2))));
    lambda(0) = std::sqrt(lambda2(0));
    lambda2_inv(0) = 1.0 / lambda2(0);
    
    psi0 = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / lambda2(0)));
    
    bl2_sum += beta2(0) * lambda2_inv(0);
    
    for (j = 1; j < p; j++)
    {
      tmp_double = (Z.row(j-1) * gamma).value();
      
      beta2(j) = std::pow(beta(j), 2);
      
      lambda(j) = rlambda(0.5 * beta2(j) / (sigma2 * tau2), 1.0 / std::sqrt(2.0 * s0phi2(j-1)), tmp_double / s0phi2(j-1));
      lambda2(j) = std::pow(lambda(j), 2);
      lambda2_inv(j) = 1.0 / lambda2(j);
      
      bl2_sum += beta2(j) * lambda2_inv(j);
      
      phi2(j-1) = 1.0 / R::rgamma(1.0, 1.0 / (0.5 * (1.0 + std::pow(lambda(j) - tmp_double, 2) / s0)));
      s0phi2(j-1) = s0 * phi2(j-1);
      s0phi2_inv(j-1) = 1.0 / s0phi2(j-1);
    }
    
    // sample gamma
    Zs = Z.transpose() * s0phi2_inv.asDiagonal();
    for (d = 0; d < D; d++)
      ds.segment(md_index(d), md(d)) = repelem(1.0 / kappa(d), md(d));
    
    gamma = rmvnorm(Zs * lambda.tail(p-1), Zs * Z, ds, s0, M);
    
    // sample kappa
    for (d = 0; d < D; d++)
      kappa(d) = 1.0 / R::rgamma(am(d), 1.0 / (b_k(d) + 0.5 * gamma.segment(md_index(d), md(d)).squaredNorm()));
    
    // sample tau2
    tau2 = 1.0 / R::rgamma(tau2_shape, 1.0 / (1.0 / eta + 0.5 * bl2_sum / sigma2));
    
    // sample eta
    eta = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / tau2));
    
    // sample sigma2
    qs = y - X * beta;
    sigma2 = 1.0 / R::rgamma(0.5 * (n + p) + v, 1.0 / (q + 0.5 * (qs.squaredNorm() + bl2_sum / tau2)));
    
    // sample beta
    beta = rmvnorm_b(X, y, tau2 * lambda2, sigma2);
    
    if (b >= bn)
    {
      Beta_out.row(b-bn) = beta.transpose();
      Lambda_out.row(b-bn) = lambda.transpose();
      Gamma_out.row(b-bn) = gamma.transpose();
      tau2_out(b-bn) = tau2;
      sigma2_out(b-bn) = sigma2;
    }
    
    if ((ping != 0) && (b % ping == 0))
      Rcpp::Rcout << "\n iter: " << b;
  }
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // output
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  return Rcpp::List::create(Rcpp::Named("Beta") = Beta_out, Rcpp::Named("Lambda") = Lambda_out, Rcpp::Named("sigma2") = sigma2_out, Rcpp::Named("tau2") = tau2_out, Rcpp::Named("Gamma") = Gamma_out);
}





// Rcpp::List infHS_probit_codata_fast (const int& B, const int& bn, const Eigen::VectorXi& y, const Eigen::MatrixXd& X, const Rcpp::List Z_list, const int& M, const Eigen::VectorXd& a_k, const Eigen::VectorXd& b_k, const double& s0 = 1, const int& ping = 1000)
// {
//   // B: number of iterations
//   // bn: number of burn-in iterations
//   // y: categorical response variable, n-dim vector
//   // X: design matrix, n x p matrix
//   // Z: list of p x m_d co-data matrices, d = 1, ..., D, sum_d m_d = M
//   // M: number of total columns in Z_list
//   // a_k, b_k: D-dimensional vector, hyperparameters for \kappa_d, d = 1, ..., D
//   // s0: optional prior scal parameter for local shrinkage parameters lambda, default equal to 1 (Horseshoe)
//   // ping: optional integer, print the number of iterations completed each "ping" iterations, default is ping = 1000
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // initialization
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   int b, i, j, d;
//   double tmp_double, bl2_sum;
//   
//   const int n = y.size(), p = X.cols(), D = a_k.size();
//   
//   Eigen::VectorXi md(D), md_index(D+1), am(D);
//   Eigen::MatrixXd Z(p, M), Zs(M, p);
//   
//   md_index(0) = 0;
//   for (d = 0; d < D; d++) 
//   {
//     Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
//     
//     md(d) = Zd.cols();
//     md_index(d+1) = md_index(d) + md(d);
//     am(d) = a_k(d) + 0.5 * (double)md(d);
//     
//     Z.block(0, md_index(d), p, md(d)) = Zd;
//   }
//   
//   Eigen::VectorXd ys(n);
//   for (i = 0; i < n; i++) 
//   {
//     if (y(i) == 1) 
//     {
//       ys(i) = ltnorm(0, 1);
//     } else 
//     {
//       ys(i) = rtnorm(0, 1);
//     }
//   }
//   
//   const double tau2_shape = 0.5 * ((double)p + 1.0);
//   double tau2 = 1.0, eta = 1.0;
//   
//   Eigen::VectorXd beta(p), lambda(p), phi2(p), gamma(M), kappa(D), ds(M);
//   Eigen::VectorXd beta2(p), lambda2(p), lambda2_inv(p);
//   lambda.setOnes(); gamma.setZero(); kappa.setOnes();
//   lambda2.setOnes(); phi2.setOnes();
//   
//   Eigen::VectorXd s0phi2_inv(p), s0phi2 = repelem(s0, p);
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // output objects
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   Eigen::VectorXd tau2_out(B-bn);
//   Eigen::MatrixXd Beta_out(B-bn, p), Lambda_out(B-bn, p), Gamma_out(B-bn, M);
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // main cycle
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   for (b = 0; b < B; b++)
//   {
//     // sample beta
//     beta = rmvnorm_b(X, ys, tau2 * lambda2, 1.0);
//     
//     // sample lambda and phi2
//     bl2_sum = 0.0;
//     for (j = 0; j < p; j++)
//     {
//       tmp_double = (Z.row(j) * gamma).value();
//       
//       beta2(j) = std::pow(beta(j), 2);
//       
//       lambda(j) = rlambda_old(0.5 * beta2(j) / tau2, 1.0 / std::sqrt(2.0 * s0phi2(j)), tmp_double / s0phi2(j));
//       lambda2(j) = std::pow(lambda(j), 2);
//       lambda2_inv(j) = 1.0 / lambda2(j);
//       
//       bl2_sum += beta2(j) * lambda2_inv(j);
//       
//       phi2(j) = 1.0 / R::rgamma(1.0, 1.0 / (0.5 + 0.5 * std::pow(lambda(j) - tmp_double, 2) / s0));
//       s0phi2(j) = s0 * phi2(j);
//       s0phi2_inv(j) = 1.0 / s0phi2(j);
//     }
//     
//     // sample gamma
//     Zs = Z.transpose() * s0phi2_inv.asDiagonal();
//     for (d = 0; d < D; d++)
//       ds.segment(md_index(d), md(d)) = repelem(1.0 / kappa(d), md(d));
//     
//     gamma = rmvnorm(Zs * lambda, Zs * Z, ds, s0, M);
//     
//     // sample kappa
//     for (d = 0; d < D; d++)
//       kappa(d) = 1.0 / R::rgamma(am(d), 1.0 / (b_k(d) + 0.5 * gamma.segment(md_index(d), md(d)).squaredNorm()));
//     
//     // sample tau2
//     tau2 = 1.0 / R::rgamma(tau2_shape, 1.0 / (1.0 / eta + 0.5 * bl2_sum));
//     
//     // sample eta
//     eta = 1.0 / R::rgamma(0.5, 1.0 / (1.0 + 1.0 / tau2));
//     
//     // update y*
//     for (i = 0; i < n; i++) 
//     {
//       if (y(i) == 1) 
//       {
//         ys(i) = ltnorm(X.row(i) * beta, 1);
//       } 
//       else 
//       {
//         ys(i) = rtnorm(X.row(i) * beta, 1);
//       }
//     }
//     
//     if (b >= bn)
//     {
//       Beta_out.row(b-bn) = beta.transpose();
//       Lambda_out.row(b-bn) = lambda.transpose();
//       Gamma_out.row(b-bn) = gamma.transpose();
//       tau2_out(b-bn) = tau2;
//     }
//     
//     if ((ping != 0) && (b % ping == 0))
//       Rcpp::Rcout << "\n iter: " << b;
//   }
//   
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   // output
//   // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//   
//   return Rcpp::List::create(Rcpp::Named("Beta") = Beta_out, Rcpp::Named("Lambda") = Lambda_out, Rcpp::Named("tau2") = tau2_out, Rcpp::Named("Gamma") = Gamma_out);
// }
// // end file
// 
