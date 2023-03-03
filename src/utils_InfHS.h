/* File: utils_InfHS.h */

#ifndef utils_InfHS_H
#define utils_InfHS_H


double rlambda (const double& d, const double& a, const double& b);

Eigen::VectorXd repelem (const double& x, const int& n);
Eigen::VectorXd rmvnorm (const Eigen::VectorXd& m, const Eigen::MatrixXd& XtX, const Eigen::VectorXd& d, const double& sigma2, const int& n); 
Eigen::VectorXd rmvnorm_b (const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& d, const double& sigma2);

#endif