/* File: utils_InfHS.h */

#ifndef utils_InfHS_old_H
#define utils_InfHS_old_H


double rlambda_old (const double& d, const double& a, const double& b);

Eigen::VectorXd repelem_old (const double& x, const int& n);
Eigen::VectorXd rmvnorm_old (const Eigen::VectorXd& m, const Eigen::MatrixXd& XtX, const Eigen::VectorXd& d, const int& n); 
Eigen::VectorXd rmvnorm_b_old (const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& d);

#endif