print(__doc__)
# Tuning the lasso parameter and show the best estimation
# Figure 4,5

from MyFunc import mylasso
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

# Lasso
lam=np.linspace(0.32,0.9) #0.25-1
rmse=np.array([])
for i in range(lam.shape[0]):
    lpart,tor=mylasso(X,lam=lam[i])
    rmse = np.append(rmse, np.linalg.norm(lpart-part,ord='fro'))

llam=lam[np.nonzero(rmse == rmse.min())]
lrmse=rmse.min()
print(llam,lrmse)
lpart,tor=mylasso(X,lam=llam)

# plot result
plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.05,right=0.98)
vmax=1
plt.subplot(1,2,1)
plt.plot(lam, rmse, 'ro-')
plt.xlim((0.2,1))
plt.xlabel('lambda\n (a)')
plt.ylabel('RSE')
plt.title('Lasso')
plt.subplot(1,2,2)
plt.imshow(lpart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Lasso Penalized Partial Correlation')
plt.xlabel('\n (b)')
plt.show()

# show the convergency
_,tor1=mylasso(X,lam=0.2)
plt.figure()
plt.subplot(1,2,1)
plt.plot(tor1,'bo-')
plt.xlabel('iterations\n (a)')
plt.ylabel('tor')
plt.title('lambda=0.2')
plt.subplot(1,2,2)
plt.title('lambda=0.6')
plt.plot(tor,'bo-')
plt.xlabel('iterations\n (b)')
plt.ylabel('tor')
plt.show()