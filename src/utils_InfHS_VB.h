/* File: utils_InfHS_VB.h */

#ifndef utils_InfHS_VB_H
#define utils_InfHS_VB_H

double mean_ig_logx (const double& a, const double& b);

// Eigen::VectorXd repelem (const double& x, const int& n);
Eigen::MatrixXd solve_largep (const Eigen::MatrixXd& X, Eigen::VectorXd& dinv);
Eigen::MatrixXd solve_smallp (const Eigen::MatrixXd& X);

// std::tuple<double, Eigen::MatrixXd> solve_largep_fastdet (const Eigen::MatrixXd& X, Eigen::VectorXd& dinv);

std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double> step_VB_det (Eigen::VectorXd& dinv, const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> step_VB_det0 (Eigen::VectorXd& dinv, const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

std::tuple<Eigen::VectorXd, Eigen::VectorXd> step_VB_pr (Eigen::VectorXd& dinv, const Eigen::MatrixXd& X, const Eigen::VectorXd& ys);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> step_VB_pr_det (Eigen::VectorXd& dinv, const Eigen::MatrixXd& X, const Eigen::VectorXd& ys);

double log_kf (const double& d, const double& a, const double& b);
double m_e1 (const double& d, const double& a, const double& b, const double& lkf);
double m_e2 (const double& d, const double& a, const double& b, const double& lkf);
double m_e3 (const double& d, const double& a, const double& b, const double& lkf);

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ihs_pstep0 (const Eigen::VectorXd& mu_beta, const Eigen::VectorXd& dSigma, const double& s0, const double& s02inv);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ihs_pstep (const Eigen::MatrixXd& Z, const Eigen::VectorXd& mu_beta, const Eigen::VectorXd& dSigma, const Eigen::VectorXd& mu_gamma, const Eigen::MatrixXd& Sigma_gamma, const Eigen::VectorXd& dphi, const Eigen::VectorXd& muphi, const double& mu_tau2, const double& mu_sigma, const double& mu_sigma2, const double& s0, const double& s02inv);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ihs_pstep_pr (const Eigen::MatrixXd& Z, const Eigen::VectorXd& mu_beta, const Eigen::VectorXd& dSigma, const Eigen::VectorXd& mu_gamma, const Eigen::MatrixXd& Sigma_gamma, const Eigen::VectorXd& dphi, const Eigen::VectorXd& muphi, const double& mu_tau2, const double& s0, const double& s02inv);

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ihs_pstep02 (const Eigen::VectorXd& mu_beta, const Eigen::VectorXd& dSigma);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ihs_pstep2 (const Eigen::MatrixXd& Z, const Eigen::VectorXd& mu_beta, const Eigen::VectorXd& dSigma, const Eigen::VectorXd& mu_gamma, const Eigen::MatrixXd& Sigma_gamma, const Eigen::VectorXd& dphi, const Eigen::VectorXd& muphi, const double& mu_tau2, const double& mu_sigma, const double& mu_sigma2);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ihs_pstep_pr2 (const Eigen::MatrixXd& Z, const Eigen::VectorXd& mu_beta, const Eigen::VectorXd& dSigma, const Eigen::VectorXd& mu_gamma, const Eigen::MatrixXd& Sigma_gamma, const Eigen::VectorXd& dphi, const Eigen::VectorXd& muphi, const double& mu_tau2);

#endif