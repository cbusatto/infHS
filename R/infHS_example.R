#library(infHS)
library(glmnet)
library(pROC)

lf = function(x, d, a, b, lkf) -log(x) - log(1 + x^2) - d / x^2 - a^2 * x^2 + b * x - lkf
tf = function(x, d, a, b, lkf, lxs) exp(-log(x) - log(1 + x^2) - d / x^2 - a^2 * x^2 + b * x - lkf - lxs)

plambda = function(d, a, b) {
  lkf = log_kf(d, a, b)
  
  xs = optimize(lf, c(0, 1000), d, a, b, lkf, maximum = T)$maximum
  lxs = lf(xs, d, a, b, lkf)
  
  i0 = integrate(tf, 0, Inf, d, a, b, lkf, lxs)$value
  return (exp(log(i0) + lxs))
}

set.seed(22)

n = 50
ntest = 30
p = 200
p0 = 50

## create design matrix
X = matrix(rnorm(n*p), n, p)
X = cbind(rep(1, n), scale(X))
X_test = matrix(rnorm(ntest*p), ntest, p)
X_test = cbind(rep(1, ntest), scale(X_test))

## sample true beta
beta0 = rep(0, p+1)
beta0[1] = rnorm(1, 0, sqrt(0.5))
u = rbinom(p0, 1, 0.4)
ix = sample(1:p, p0)
ix_no = setdiff(1:p, ix)
beta0[ix+1] = (-1)^(u) * (0.75 * log(n) / sqrt(n) + abs(rnorm(p0, 0, sqrt(0.75))))

## true model
true_model = rep(0, p+1)
true_model[1] = 1
true_model[ix+1] = 1

## response variable
y = X %*% beta0 + rnorm(n)
y_test = X_test %*% beta0 + rnorm(ntest)

## :::: SIMULATE CO-DATA SOURCES (D = 2) ::::

p01 = 30
p02 = 10
G1size = 50
G2size = 70

G1 = rep(0, p)
G1[ix[1:5]] = 1
G1[sample(ix[-c(1:5)], p01-5)] = 1
G1[sample(ix_no, G1size-p01)] = 1

G2 = rep(0, p)
G2[ix[1:5]] = 1
G2[sample(ix[-c(1:5)], p02-5)] = 1
G2[sample(ix_no, G2size-p02)] = 1

Z = list()
Z[[1]] = rep(1, p)
Z[[2]] = model.matrix(~ -1 + as.factor(G1))[, 2]
Z[[3]] = model.matrix(~ -1 + as.factor(G2))[, 2]

D = 2 + 1   ## (intercept included)
md = c(1, 1, 1)

## :::: SET HYPERPARAMETERS ::::

## hyperparameters for sigma^2
hyp_sigma = c(0.5, 10)

## hyperparameters for \kappa_d
a_k = c(1, 1, 1)
b_k = c(5, 10, 10)

## ::::::::::::::::
## RUN infHS GIBBS

B = 10000
bn = 5000

res_infHS_GIBBS = infHS_FB(B, bn, 
                           y = y, X = X, Z = Z, M = sum(md), 
                           hyp_sigma = hyp_sigma, 
                           a_k = a_k, b_k = b_k, 
                           ping = 1000)

## posterior inference

## Estimated beta and sd
beta_FB = apply(res_infHS_GIBBS$Beta, 2, mean)
sd_FB = apply(res_infHS_GIBBS$Beta, 2, sd)

## Empirical coverage rate of CI
CI_FB = apply(res_infHS_GIBBS$Beta[, -1], 2, function(x) quantile(x, probs = c(0.025, 0.975)))
cov_FB = (CI_FB[1, ] <= beta0[-1]) & (beta0[-1] <= CI_FB[2, ])
width_FB = CI_FB[2, ] - CI_FB[1, ]

## RMSE0 and RMSE1 for beta
rmse1_FB = sqrt(mean((beta0[-1][ix] - beta_FB[-1][ix])^2))
rmse0_FB = sqrt(mean((beta0[-1][-ix] - beta_FB[-1][-ix])^2))

## Posterior inclusion probabilies for MPM model
prob_FB = apply(res_infHS_GIBBS$Lambda[, -1], 2, function(x) mean(x / (1 + x)))

## AUC
auc_FB_mpm = pROC::roc(true_model[-1], prob_FB)$auc

## RMSE for predictive performance with Bayesian Model Averaging
pred_FB = apply(res_infHS_GIBBS$Beta, 1, function(x) X_test %*% x)
pred_FB = apply(pred_FB, 1, mean)
rmse_FB_BMA = sqrt(mean((y_test - pred_FB)^2))


## ::::::::::::::::
## RUN infHS VB

bmax = 2000

time_VB = proc.time()
res_infHS_VB = infHS_VB(y = y, X = X, Z = Z, M = sum(md), 
                        hyp_sigma = hyp_sigma,
                        a_k = a_k, b_k = b_k, 
                        eps = 0.001, ping = 250, bmax = bmax)

## Estimated beta
beta_VB = res_infHS_VB$beta
sd_VB = sqrt(res_infHS_VB$var_beta)

## Empirical coverage rate of CI
CI_VB = matrix(0, 2, p)
CI_VB[1, ] = beta_VB[-1] - 1.96 * sd_VB[-1]
CI_VB[2, ] = beta_VB[-1] + 1.96 * sd_VB[-1]
cov_VB = (CI_VB[1, ] <= beta0[-1]) & (beta0[-1] <= CI_VB[2, ])
width_VB = CI_VB[2, ] - CI_VB[1, ]

## RMSE0 and RMSE1 for beta
rmse1_VB = sqrt(mean((beta0[-1][ix] - beta_VB[-1][ix])^2))
rmse0_VB = sqrt(mean((beta0[-1][-ix] - beta_VB[-1][-ix])^2))

## Posterior inclusion probabilies for MPM model
prob_VB = apply(res_infHS_VB$Lambda, 1, function(x) 1 - plambda(x[1], sqrt(x[2]), x[3]))

## AUC with DSS and MPM selections  
auc_VB_mpm = pROC::roc(true_model[-1], prob_VB)$auc

## RMSE for predictive performance
pred_VB = X_test %*% beta_VB
rmse_VB = sqrt(mean((y_test - pred_VB)^2))


