print(__doc__)
# Figure 2
# Generate data and show the ground truth covariance, precision, partial correlation matrixes

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# generate data
n_samples=15
n_features=20

prng=np.random.RandomState(5)
prec=make_sparse_spd_matrix(n_features, alpha= .95,
                            smallest_coef=.4,
                            largest_coef=.7,
                            random_state=prng)

cov=linalg.inv(prec)
d=np.sqrt(np.diag(cov))
cov_=cov
cov_ /=d
cov_ /=d[:,np.newaxis]
prec_=prec
prec_ *=d
prec_ *=d[:,np.newaxis]
X=prng.multivariate_normal(np.zeros(n_features),cov,size=n_samples)
X-=X.mean(axis=0)
X/=X.std(axis=0)

X_test=prng.multivariate_normal(np.zeros(n_features),cov,size=n_samples)
X_test-=X_test.mean(axis=0)
X_test/=X_test.std(axis=0)

part=np.zeros((n_features,n_features))
for i in range(n_features):
    for j in range(n_features):
        part[i,j]=-prec[i,j]/np.sqrt(prec[i,i]*prec[j,j])

np.savetxt("data/data.csv",X,delimiter=",")
np.savetxt("data/test_data.csv",X_test,delimiter=",")
np.savetxt("data/true_cov.csv",cov,delimiter=",")
np.savetxt("data/true_prec.csv",prec,delimiter=",")
np.savetxt("data/est_cov.csv",cov_,delimiter=",")
np.savetxt("data/est_prec.csv",prec_,delimiter=",")

# plot
plt.figure(figsize=(10,6))
plt.subplots_adjust(left=0.02,right=0.98)
covs=[('True Covariance',cov),('True Precision',prec),('True Partial Correlation',part)]
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(1,3,i+1)
    vmax=vmax=np.maximum(this_cov.max(),abs(this_cov.min()))
    plt.imshow(this_cov,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
    plt.colorbar()
    plt.title(name)
plt.show()