#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppNumerical.h>
#include <math.h>     

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]

#include "utils_InfHS.h"
#include "utils_InfHS_VB.h"


// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// LINEAR REGRESSION
// Variational bayes approximation to the posterior
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

// [[Rcpp::export]]

Rcpp::List infHS_VB (const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Rcpp::List Z_list, const int& M, const Eigen::VectorXd& hyp_sigma, const Eigen::VectorXd& a_k, const Eigen::VectorXd& b_k, const int& bmax = 2000, const double& eps = 1e-03, const int& ping = 100)
{
  // y: numeric response variable, n-dim vector
  // X: design matrix, n x (p+1) matrix
  // Z_list: list of p x m_d co-data matrices, d = 1, ..., D, sum_d m_d = M
  // M: number of total columns in Z_list
  // hyp_sigma: hyperparameters for sigma2
  // a_k, b_k: D-dimensional vector, hyperparameters for \kappa_d, d = 1, ..., D
  // bmax: maximum number of iterations, default is bmax = 1000
  // ping: optional integer, print the number of iterations completed each "ping" iterations, default is ping = 1000
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // initialization
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  const int n = y.size(), p = X.cols(), D = a_k.size();
  int b = 0, j, d, m;
  
  const double p12 = 0.5 * (double)p, w_sigma2 = hyp_sigma(0) + 0.5 * double(n + p - 1) + 1.0;
  double lb1 = 1.0, lb2 = 0.0, tmp_d;
  
  Eigen::VectorXi md(D), md_index(D+1);
  Eigen::MatrixXd Z(p-1, M);
  
  md_index(0) = 0;
  for (d = 0; d < D; d++) 
  {
    Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
    md(d) = Zd.cols();
    md_index(d+1) = md_index(d) + md(d);
    
    Z.block(0, md_index(d), p-1, md(d)) = Zd;
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize beta
  
  double ldet_beta, bl_sum0 = 0.0;
  Eigen::VectorXd mu_beta(p), dSigma(p), mu_lambda(p);
  mu_beta.setOnes();
  
  std::tie(mu_beta, dSigma, ldet_beta) = step_VB_det0(mu_beta, X, y);
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize lambda0
  
  double mu_beta2 = std::pow(mu_beta(0), 2.0) + dSigma(0);
  double a0 = 1.0 + 0.5 * mu_beta2;
  
  double mu_lambda0 = 1.0 / a0;
  bl_sum0 = mu_beta2 / a0;
  
  double k0 = 1.0 + mu_lambda0;
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize lambda1, .., lambdap
  
  // double lkf_p, lkf, e1, e2, e3;
  Eigen::VectorXd a_lambda(p-1), b_lambda(p-1), c_lambda(p-1), d_phi(p-1);
  Eigen::VectorXd lambda_phi(p-1), mu_phi(p-1), bl_sum(p-1), tmp_j(p-1);
  
  std::tie(a_lambda, b_lambda, c_lambda, d_phi, mu_lambda, mu_phi, lambda_phi, bl_sum, tmp_j) = ihs_pstep02(mu_beta, dSigma);
  mu_lambda(0) = mu_lambda0;
  
  double tmp_j0 = 0.0, bl_sumj = 0.0;
  for (j = 0; j < p-1; j++)
  {
    bl_sumj += bl_sum(j);
    tmp_j0 += tmp_j(j);
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize gamma
  
  Eigen::VectorXd mu_kappa(M);
  for (d = 0; d < D; d++)
    mu_kappa.segment(md_index(d), md(d)) = repelem(a_k(d) / b_k(d), md(d));
  
  Eigen::MatrixXd Zs = Z.transpose() * mu_phi.asDiagonal();
  Eigen::MatrixXd ZsZ = Zs * Z;
  ZsZ.diagonal() += mu_kappa;
  Eigen::MatrixXd Sigma_gamma = solve_smallp(ZsZ);
  Eigen::VectorXd mu_gamma = Sigma_gamma * (Z.transpose() * lambda_phi);
  
  double ldet_gamma = std::log(Sigma_gamma.determinant());
  
  Eigen::VectorXd e_kappa(D), f_kappa(D);
  tmp_d = 0.0;
  for (d = 0; d < D; d++)
  {
    e_kappa(d) = a_k(d) + 0.5 * md(d);
    f_kappa(d) = b_k(d);
    for (m = md_index(d); m < md_index(d) + md(d); m++)
      f_kappa(d) += 0.5 * (std::pow(mu_gamma(m), 2.0) + Sigma_gamma(m, m));
    
    tmp_d -= e_kappa(d) * std::log(f_kappa(d));
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize tau2
  
  double g_tau2 = 0.5 * bl_sumj + 1.0;
  double mu_tau2 = p12 / g_tau2;
  
  double h_zeta = 1.0 + mu_tau2;
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize sigma2 and update beta
  
  double trace_xSx;
  Eigen::VectorXd res = y - X * mu_beta;
  
  mu_lambda(0) *= 1.0 / mu_tau2;
  mu_beta = mu_tau2 * mu_lambda;
  std::tie(mu_beta, dSigma, ldet_beta, trace_xSx) = step_VB_det(mu_beta, X, y);
  
  double l_sigma2 = hyp_sigma(1) + 0.5 * res.squaredNorm() + 0.5 * trace_xSx + 0.5 * bl_sum0 + 0.5 * bl_sumj * mu_tau2 + 1.0 / h_zeta;
  double mu_sigma2 = w_sigma2 / l_sigma2;
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize ELBO 
  
  lb2 = tmp_j0 + tmp_d + 0.5 * (ldet_beta + ldet_gamma) - p12 * std::log(g_tau2) - std::log(h_zeta) - w_sigma2 * std::log(l_sigma2) + 0.5 * p * mean_ig_logx(w_sigma2, l_sigma2) - std::log(a0) - std::log(k0) + w_sigma2 / (h_zeta * l_sigma2);
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // start algorithm
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  while ((std::abs(lb2 - lb1) > eps) && (b < bmax))
  {
    lb1 = lb2;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update lambda0
    
    mu_beta2 = std::pow(mu_beta(0), 2.0) + dSigma(0) * l_sigma2 / (w_sigma2 - 1.0);
    a0 = 1.0 / k0 + 0.5 * mu_beta2 * mu_sigma2;
    
    mu_lambda0 = 1.0 / a0;
    bl_sum0 = mu_beta2 * mu_lambda0;
    
    k0 = 1.0 + mu_lambda0;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // parallel update lambda1, ..., lambdap
    
    std::tie(a_lambda, b_lambda, c_lambda, d_phi, mu_lambda, mu_phi, lambda_phi, bl_sum, tmp_j) = ihs_pstep2(Z, mu_beta, dSigma, mu_gamma, Sigma_gamma, d_phi, mu_phi, mu_tau2, l_sigma2 / (w_sigma2 - 1.0), mu_sigma2);
    mu_lambda(0) = mu_lambda0;
    
    tmp_j0 = 0.0; bl_sumj = 0.0;
    for (j = 0; j < p-1; j++)
    {
      bl_sumj += bl_sum(j);
      tmp_j0 += tmp_j(j);
    }
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update gamma
    
    for (d = 0; d < D; d++)
      mu_kappa.segment(md_index(d), md(d)) = repelem(e_kappa(d) / f_kappa(d), md(d));
    
    Zs = Z.transpose() * mu_phi.asDiagonal();
    ZsZ = Zs * Z;
    ZsZ.diagonal() += mu_kappa;
    Sigma_gamma = solve_smallp(ZsZ);
    mu_gamma = Sigma_gamma * (Z.transpose() * lambda_phi);
    
    ldet_gamma = std::log(Sigma_gamma.determinant());
    
    tmp_d = 0.0;
    for (d = 0; d < D; d++)
    {
      f_kappa(d) = b_k(d);
      for (m = md_index(d); m < md_index(d) + md(d); m++)
        f_kappa(d) += 0.5 * (std::pow(mu_gamma(m), 2.0) + Sigma_gamma(m, m));
      
      tmp_d -= e_kappa(d) * std::log(f_kappa(d));
    }
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update tau2
    
    g_tau2 = 0.5 * bl_sumj * mu_sigma2 + 1.0 / h_zeta;
    mu_tau2 = p12 / g_tau2;
    
    h_zeta = mu_sigma2 + mu_tau2;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update sigma2 and beta
    
    res = y - X * mu_beta;
    mu_lambda(0) *= 1.0 / mu_tau2;
    mu_beta = mu_tau2 * mu_lambda;
    std::tie(mu_beta, dSigma, ldet_beta, trace_xSx) = step_VB_det(mu_beta, X, y);
    
    l_sigma2 = hyp_sigma(1) + 0.5 * (res.squaredNorm() + trace_xSx + bl_sum0 + bl_sumj * mu_tau2) + 1.0 / h_zeta;
    
    mu_sigma2 = w_sigma2 / l_sigma2;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update ELBO
    
    lb2 = tmp_j0 + tmp_d + 0.5 * (ldet_beta + ldet_gamma) - p12 * std::log(g_tau2) - std::log(h_zeta) - w_sigma2 * std::log(l_sigma2) + 0.5 * p * mean_ig_logx(w_sigma2, l_sigma2) - std::log(a0) - std::log(k0) + w_sigma2 / (h_zeta * l_sigma2);
    
    if ((ping != 0) && (b % ping == 0))
    {
      Rcpp::Rcout << "\n iter: " << b;
      // Rcpp::Rcout << "\n ELBO: " << lb2;
      // Rcpp::Rcout << "\n ELBO increase: " << lb2 - lb1;
    }
    
    b += 1;
  }
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // output
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  // if (b >= bmax)
  // {
  //   Rcpp::Rcout << "\n Warning: algorithm did not converge in " << bmax << " iterations.";
  //   Rcpp::Rcout << "\n " << std::abs(lb2 - lb1) << "\n";
  // } 
  // else
  // {
  //   Rcpp::Rcout << "\n ********** Elbo difference: *********";
  //   Rcpp::Rcout << "\n " << std::abs(lb2 - lb1) << "\n";
  // }
  
  Eigen::MatrixXd lambda_out(p-1, 3);
  lambda_out.col(0) = a_lambda;
  lambda_out.col(1) = b_lambda;
  lambda_out.col(2) = c_lambda;
  
  return Rcpp::List::create(Rcpp::Named("beta") = mu_beta, Rcpp::Named("var_beta") = dSigma, Rcpp::Named("lambda0") = a0, Rcpp::Named("Lambda") = lambda_out, Rcpp::Named("tau2") = g_tau2 / (p12 - 1.0), Rcpp::Named("sigma2") = l_sigma2 / (w_sigma2 - 1.0), Rcpp::Named("gamma") = mu_gamma, Rcpp::Named("Sigma_gamma") = Sigma_gamma, Rcpp::Named("phi") = d_phi);
}



// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// PROBIT REGRESSION following Albert & Chib (1993)
// Variational bayes approximation to the augmented posterior
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

// [[Rcpp::export]]

Rcpp::List infHS_VB_probit (const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Rcpp::List Z_list, const int& M, const Eigen::VectorXd& a_k, const Eigen::VectorXd& b_k, const int& bmax = 2000, const double& eps = 1e-03, const int& ping = 100)
{
  // y: binary response variable, n-dim vector
  // X: design matrix, n x (p+1) matrix
  // Z_list: list of p x m_d co-data matrices, d = 1, ..., D, sum_d m_d = M
  // M: number of total columns in Z_list
  // hyp_sigma: hyperparameters for sigma2
  // a_k, b_k: D-dimensional vector, hyperparameters for \kappa_d, d = 1, ..., D
  // bmax: maximum number of iterations, default is bmax = 1000
  // ping: optional integer, print the number of iterations completed each "ping" iterations, default is ping = 1000
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // initialization
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  const int n = y.size(), p = X.cols(), D = a_k.size();
  int b = 0, i, j, d, m;
  
  const double s02inv = 0.5, p12 = 0.5 * (double)p;
  double lb1 = 1.0, lb2 = 0.0, tmp_d;
  
  Eigen::VectorXi md(D), md_index(D+1);
  Eigen::MatrixXd Z(p-1, M);
  
  md_index(0) = 0;
  for (d = 0; d < D; d++) 
  {
    Eigen::Map<Eigen::MatrixXd> Zd(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Z_list[d]));
    md(d) = Zd.cols();
    md_index(d+1) = md_index(d) + md(d);
    
    Z.block(0, md_index(d), p-1, md(d)) = Zd;
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize ys
  
  Eigen::VectorXd ys(n);
  double tmp_i = n * std::log(0.5);
  for (i = 0; i < n; i++) 
  {
    if (y(i) == 1) 
    {
      ys(i) = 0.7978846;
    } 
    else 
    {
      ys(i) = -0.7978846;
    }
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize beta
  
  double ldet_beta, bl_sum0 = 0.0;
  Eigen::VectorXd mu_beta(p), dSigma(p), mu_lambda(p);
  mu_beta.setOnes();
  
  std::tie(mu_beta, dSigma, ldet_beta) = step_VB_det0(mu_beta, X, y);
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize lambda0
  
  double mu_beta2 = std::pow(mu_beta(0), 2.0) + dSigma(0);
  double a0 = 1.0 + 0.5 * mu_beta2;
  
  double mu_lambda0 = 1.0 / a0;
  bl_sum0 = mu_beta2 / a0;
  
  double k0 = 1.0 + mu_lambda0;
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize lambda1, .., lambdap
  
  // double lkf_p, lkf, e1, e2, e3;
  Eigen::VectorXd a_lambda(p-1), b_lambda(p-1), c_lambda(p-1), d_phi(p-1);
  Eigen::VectorXd lambda_phi(p-1), mu_phi(p-1), bl_sum(p-1), tmp_j(p-1);
  
  std::tie(a_lambda, b_lambda, c_lambda, d_phi, mu_lambda, mu_phi, lambda_phi, bl_sum, tmp_j) = ihs_pstep02(mu_beta, dSigma);
  mu_lambda(0) = mu_lambda0;
  
  double tmp_j0 = 0.0, bl_sumj = 0.0;
  for (j = 0; j < p-1; j++)
  {
    bl_sumj += bl_sum(j);
    tmp_j0 += tmp_j(j);
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize gamma
  
  Eigen::VectorXd mu_kappa(M);
  for (d = 0; d < D; d++)
    mu_kappa.segment(md_index(d), md(d)) = repelem(a_k(d) / b_k(d), md(d));
  
  Eigen::MatrixXd Zs = Z.transpose() * mu_phi.asDiagonal();
  Eigen::MatrixXd ZsZ = Zs * Z;
  ZsZ.diagonal() += mu_kappa;
  Eigen::MatrixXd Sigma_gamma = solve_smallp(ZsZ);
  Eigen::VectorXd mu_gamma = Sigma_gamma * (Z.transpose() * lambda_phi);
  
  double ldet_gamma = std::log(Sigma_gamma.determinant());
  
  Eigen::VectorXd e_kappa(D), f_kappa(D);
  tmp_d = 0.0;
  for (d = 0; d < D; d++)
  {
    e_kappa(d) = a_k(d) + 0.5 * md(d);
    f_kappa(d) = b_k(d);
    for (m = md_index(d); m < md_index(d) + md(d); m++)
      f_kappa(d) += 0.5 * (std::pow(mu_gamma(m), 2.0) + Sigma_gamma(m, m));
    
    tmp_d -= e_kappa(d) * std::log(f_kappa(d));
  }
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize tau2
  
  double g_tau2 = 0.5 * bl_sumj + 1.0;
  double mu_tau2 = p12 / g_tau2;
  
  double h_zeta = 1.0 + p12 / g_tau2;
  
  // ::::::::::::::::::::::::::::::::::::::::::::::::
  // initialize ELBO 
  
  lb2 = tmp_i + tmp_j0 + tmp_d + 0.5 * (ldet_beta + ldet_gamma) - p12 * std::log(g_tau2) - std::log(h_zeta) - std::log(a0) - std::log(k0);
  lb1 = lb2 + 1.0;
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // start algorithm
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  while ((std::abs(lb2 - lb1) > eps) && (b < bmax))
  {
    lb1 = lb2;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update ys
    
    tmp_i = 0.0;
    for (i = 0; i < n; i++) 
    {
      double mi = X.row(i) * mu_beta;
      double pi = R::pnorm(-mi, 0.0, 1.0, true, false);
      
      if (y(i) == 1) 
      {
        ys(i) = mi + R::dnorm(-mi, 0.0, 1.0, false) / (1.0 - pi);
        tmp_i += std::log(1.0 - pi);
      } 
      else 
      {
        ys(i) = mi - R::dnorm(-mi, 0.0, 1.0, false) / pi;
        tmp_i += std::log(pi);
      }
    }
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update beta
    
    mu_lambda(0) *= 1.0 / mu_tau2;
    mu_beta = mu_tau2 * mu_lambda;
    std::tie(mu_beta, dSigma, ldet_beta) = step_VB_pr_det(mu_beta, X, ys);
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update lambda0
    
    mu_beta2 = std::pow(mu_beta(0), 2.0) + dSigma(0);
    a0 = 1.0 / k0 + 0.5 * mu_beta2 * mu_tau2;
    
    mu_lambda0 = 1.0 / a0;
    bl_sum0 = mu_beta2 / a0;
    
    k0 = 1.0 + mu_lambda0;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // parallel update lambda1, ..., lambdap
    
    std::tie(a_lambda, b_lambda, c_lambda, d_phi, mu_lambda, mu_phi, lambda_phi, bl_sum, tmp_j) = ihs_pstep_pr2(Z, mu_beta, dSigma, mu_gamma, Sigma_gamma, d_phi, mu_phi, mu_tau2);
    mu_lambda(0) = mu_lambda0;
    
    tmp_j0 = 0.0; bl_sumj = 0.0;
    for (j = 0; j < p-1; j++)
    {
      bl_sumj += bl_sum(j);
      tmp_j0 += tmp_j(j);
    }
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update gamma
    
    for (d = 0; d < D; d++)
      mu_kappa.segment(md_index(d), md(d)) = repelem(e_kappa(d) / f_kappa(d), md(d));
    
    Zs = Z.transpose() * mu_phi.asDiagonal();
    ZsZ = Zs * Z;
    ZsZ.diagonal() += mu_kappa;
    Sigma_gamma = solve_smallp(ZsZ);
    mu_gamma = Sigma_gamma * (Z.transpose() * lambda_phi);
    
    ldet_gamma = std::log(Sigma_gamma.determinant());
    
    tmp_d = 0.0;
    for (d = 0; d < D; d++)
    {
      f_kappa(d) = b_k(d);
      for (m = md_index(d); m < md_index(d) + md(d); m++)
        f_kappa(d) += 0.5 * (std::pow(mu_gamma(m), 2.0) + Sigma_gamma(m, m));
      
      tmp_d -= e_kappa(d) * std::log(f_kappa(d));
    }
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update tau2
    
    g_tau2 = 0.5 * bl_sumj + 1.0 / h_zeta;
    mu_tau2 = p12 / g_tau2;
    
    h_zeta = 1.0 + p12 / g_tau2;
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::
    // update ELBO
    
    lb2 = tmp_i + tmp_j0 + tmp_d + 0.5 * (ldet_beta + ldet_gamma) - p12 * std::log(g_tau2) - std::log(h_zeta) - std::log(a0) - std::log(k0);
    
    
    if ((ping != 0) && (b % ping == 0))
    {
      Rcpp::Rcout << "\n iter: " << b;
      // Rcpp::Rcout << "\n ELBO: \n" << lb2;
      // Rcpp::Rcout << "\n ELBO increase: \n" << lb2 - lb1;
    }
    
    b += 1;
  }
  
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  // output
  // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  
  // if (b >= bmax)
  // {
  //   Rcpp::Rcout << "\n Warning: algorithm did not converge";
  //   Rcpp::Rcout << "\n " << lb2 - lb1;
  // } 
  // else
  // {
  //   Rcpp::Rcout << "\n ********** Elbo difference: *********";
  //   Rcpp::Rcout << "\n " << lb2 - lb1;
  // }
  
  Eigen::MatrixXd lambda_out(p-1, 3);
  lambda_out.col(0) = a_lambda;
  lambda_out.col(1) = b_lambda;
  lambda_out.col(2) = c_lambda;
  
  return Rcpp::List::create(Rcpp::Named("beta") = mu_beta, Rcpp::Named("var_beta") = dSigma, Rcpp::Named("lambda0") = a0, Rcpp::Named("Lambda") = lambda_out, Rcpp::Named("tau2") = g_tau2 / (p12 - 1.0), Rcpp::Named("gamma") = mu_gamma, Rcpp::Named("Sigma_gamma") = Sigma_gamma);
}
