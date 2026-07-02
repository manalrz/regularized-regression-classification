###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
###############################################################################

################################################################################
# PARAMETERS
################################################################################
# Dimension and sample size
p=2
n=600
# Proportion of sample from classes 0, 1, and outliers
p0 = 1/6
p1 = 1/6
pout = 4/6
# Examples of means/covariances of classes 0, 1 and outliers
mu0 = np.array([-2,-2])
mu1 = np.array([2,2])
muout = np.array([-8,-8])
Sigma_ex1 = np.eye(p)
Sigma_ex2 = np.array([[5, 0.1],
                      [1, 0.5]])
Sigma_ex3 = np.array([[0.5, 1],
                      [1, 5]])
Sigma0 = Sigma_ex1
Sigma1 = Sigma_ex1
Sigmaout = Sigma_ex1
# Regularization coefficient
lamb = 0
################################################################################

################################################################################
# DATA/LABELS GENERATION
################################################################################
# Sample sizes
n0 = int(n*p0)
n1 = int(n*p1)
nout = int(n*pout)
# Data and labels
mu0_mat = mu0.reshape((p,1))@np.ones((1,n0))
mu1_mat = mu1.reshape((p,1))@np.ones((1,n1))
x0 = np.zeros((p,n0+nout))
x0[:,0:n0] = mu0_mat + la.sqrtm(Sigma0)@rnd.randn(p,n0)
x1 = mu1_mat + la.sqrtm(Sigma1)@rnd.randn(p,n1)
if nout > 0:
  muout_mat = muout.reshape((p,1))@np.ones((1,nout))
  x0[:,n0:n0+nout] = muout_mat + la.sqrtm(Sigmaout)@rnd.randn(p,nout)
y = np.concatenate((-np.ones(n0+nout),np.ones(n1)))
X = np.ones((n,p+1))
for i in np.arange(n):
     X[0:n0+nout,1:p+1] = x0.T
     X[n0+nout:n,1:p+1] = x1.T
################################################################################
# 
################################################################################
# OLS
beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)

# ridge
lambd = 1.0
I = np.eye(X.shape[1])
I[0, 0] = 0
beta_ridge = np.linalg.inv(X.T @ X + n * lambd * I) @ (X.T @ y)

# logistique
logistic_clf = LogisticRegression()
logistic_clf.fit(X, y)
beta_logistic = np.append(logistic_clf.intercept_, logistic_clf.coef_.flatten())

# Tracer les hyperplans
x_vals = np.linspace(-10, 10, 100)
y_vals_ols = -(beta_ols[0] + beta_ols[1] * x_vals) / beta_ols[2]
y_vals_ridge = -(beta_ridge[0] + beta_ridge[1] * x_vals) / beta_ridge[2]
y_vals_logistic = -(beta_logistic[0]  + beta_logistic[1] * x_vals) / beta_logistic[2]

################################################################################
# PLOTS
################################################################################
fig,ax = plt.subplots()
ax.plot(x0[0,:],x0[1,:],'xb',label='Class 0')
ax.plot(x1[0,:],x1[1,:],'xr',label="Class 1")
ax.legend(loc = "upper left")
# Hyperplans
#ax.plot(x_vals, y_vals_ols, label='Hyperplan OLS', color='green')
#ax.plot(x_vals, y_vals_ridge, label='Hyperplan Ridge', color='purple')
ax.plot(x_vals, y_vals_logistic, label='Hyperplan Logistique', color='orange')
ax.legend(loc="upper left")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.title("Classification : OLS et Ridge")
plt.xlabel("X1")
plt.ylabel("X2")

plt.show()

# influence du pout sur l'hyperplan 

# question 3 : la fonction de perte logistique a un maximum, et ne pourra donc jamais avoir des valeurs plus grande que log(2), 
# alors que la fonction de perte quadratique dépend de la diff entre u et v et peut donc prendre des valeurs très grandes.

# question 4 : À vue d'oeil, l'hyperplan logiqtique semble moins efficace que les hyperplans OLS et Ridge.

# quetion 5 : Lorsque les outliers augmentent, l'hyperplan logistique montre des résultats bien plus concluants 
# que les hyperplans OLS et Ridge. Car l'hyperplan logistique est beaucoup moins influencé par les outliers que 
# les autres hyperplans.

