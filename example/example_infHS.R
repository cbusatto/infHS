# require(infHS)

lf = function(x, d, a, b, lkf) -log(x) - log(1 + x^2) - d / x^2 - a^2 * x^2 + b * x - lkf
tf = function(x, d, a, b, lkf, lxs) exp(-log(x) - log(1 + x^2) - d / x^2 - a^2 * x^2 + b * x - lkf - lxs)

plambda = function(d, a, b) {
  lkf = log_kf(d, a, b)
  
  xs = optimize(lf, c(0, 1000), d, a, b, lkf, maximum = T)$maximum
  lxs = lf(xs, d, a, b, lkf)
  
  i0 = integrate(tf, 0, Inf, d, a, b, lkf, lxs)$value
  return (exp(log(i0) + lxs))
}

# set values of n, p and p0 in the simulation study
n = 100
p = 250
p0 = 15

# simulate beta, X and y
X = matrix(rnorm(n*p), n, p)
X = cbind(rep(1, n), scale(X))

beta0 = rep(0, p+1)
beta0[1] = rnorm(1, 0, sqrt(0.5))
u = rbinom(p0, 1, 0.4)
beta0[2:(p0+1)] = (-1)^(u) * (0.75 * log(n) / sqrt(n) + abs(rnorm(p0, 0, sqrt(0.75))))

y = X %*% beta0 + rnorm(n)

# true model
tmodel = rep(0, p+1)
tmodel[1:(p0+1)] = 1

# simulate 2 co-data sources
Z = list()

G = rep(0, p)
G[sample(1:p0, 5)] = 1
G[sample((p0+1):p, 10)] = 1
Z[[1]] = model.matrix(~ -1 + as.factor(G))

G = rep(0, p)
G[sample(1:p, 25)] = 1
G[sample(1:p, 25)] = 2
Z[[2]] = model.matrix(~ -1 + as.factor(G))

# run infHS: Variational approximation
D = 2
md = c(ncol(Z[[1]]), ncol(Z[[2]]))

ihs = infHS::infHS_VB(y = y, X = X, Z = Z, M = sum(md), 
               hyp_sigma = c(1, 10), a_k = rep(1, D), b_k = rep(10, D), 
               eps = 0.001, ping = 500, bmax = 1000)

# get beta coefficients
beta_VB = ihs$beta

# get co-data coefficients
gamma_VB = ihs$gamma

# posterior marginal inclusion probabilities: MPM selection
pl = rep(0, p)
for (h in 1:p) {
  pl[h] = 1 - plambda(ihs$Lambda[h, 1], sqrt(ihs$Lambda[h, 2]), ihs$Lambda[h, 3])
}

pROC::roc(tmodel[-1], pl)$auc

# DSS selection
pf <- 1/abs(beta_VB)
dss = glmnet::cv.glmnet(x = X, y = X %*% beta_VB, family = "gaussian", intercept=FALSE, alpha = 1, 
                standardize = FALSE, penalty.factor = pf)


# run infHS: Gibbs sampler (to be used only on problems with moderate dimensions)
# ihs = infHS_FB(5000, 2500, y = y, X = X, Z = Z, M = sum(md),
#                hyp_sigma = c(1, 10), a_k = rep(1, D), b_k = rep(10, D), ping = 1000)
# 
# get beta coefficients
# beta_FB = apply(ihs$Beta, 2, mean)
# 
# posterior marginal inclusion probabilities: MPM selection 
# pl = apply(ihs$Lambda[, -1], 2, function(x) mean(x^2 / (1 + x^2)))

