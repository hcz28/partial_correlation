print(__doc__)
# Tuning the graphical lasso parameter and show the best estimation
# Figure 3

from MyFunc import myglasso
import numpy as np
import matplotlib.pyplot as plt

# load data
X=np.genfromtxt('data/data.csv',delimiter=',')
cov=np.genfromtxt('data/true_cov.csv',delimiter=',')
prec=np.genfromtxt('data/true_prec.csv',delimiter=',')
n_samples,n_features=X.shape
part=np.zeros((n_features,n_features))
for i in range(n_features):
    for j in range(n_features):
        part[i,j]=-prec[i,j]/np.sqrt(prec[i,i]*prec[j,j])

# Graphical lasso
lam=np.linspace(0.01, 1)
rmse=np.array([])
for i in range(lam.shape[0]):
    gpart,gprec,gcov=myglasso(X, lam=lam[i])
    rmse = np.append(rmse, np.linalg.norm(gpart - part, ord='fro'))
glam=lam[np.nonzero(rmse == rmse.min())]
grmse=rmse.min()
gpart,gprec,gcov=myglasso(X, lam=glam)
print(glam, grmse)

# plot
plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.05,right=0.98)
vmax=1
plt.subplot(1,2,1)
plt.plot(lam, rmse, 'ro-')
plt.xlabel('lambda\n (a)')
plt.ylabel('RSE')
plt.title('Graphical Lasso')

plt.subplot(1,2,2)
plt.imshow(gpart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.xlabel('\n (b)')
plt.title('Graphical Lasso Partial Correlation')
plt.show()