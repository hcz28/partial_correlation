print(__doc__)
# Tuning the lasso parameter and show the best estimation
# Figure 6

from MyFunc import myelastic
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# load data
X=np.genfromtxt('data/data.csv',delimiter=',')
cov=np.genfromtxt('data/true_cov.csv',delimiter=',')
prec=np.genfromtxt('data/true_prec.csv',delimiter=',')
n_samples,n_features=X.shape
part=np.zeros((n_features,n_features))
for i in range(n_features):
    for j in range(n_features):
        part[i,j]=-prec[i,j]/np.sqrt(prec[i,i]*prec[j,j])

alpha=np.linspace(0.1,0.5,num=20)#0.28
lam=np.linspace(1,1.8,num=20) #1.27
rmse=np.empty([lam.shape[0],alpha.shape[0]])
for i in range(lam.shape[0]):
    for j in range(alpha.shape[0]):
        epart, tor=myelastic(X, lam=lam[i], alpha=alpha[j])
        # rmse[i,j]=mean_squared_error(epart, part) ** 0.5
        rmse[i,j]=np.linalg.norm(epart-part,ord='fro')

loc=np.array(np.nonzero(rmse==rmse.min()))
elam=lam[loc[0]]
ealpha=alpha[loc[1]]
ermse=rmse.min()
print(lam[loc[0]],alpha[loc[1]],rmse.min())
epart, tor=myelastic(X, lam=1.26,alpha=0.27)

fig=plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.05,right=0.98)
X,Y=np.meshgrid(alpha,lam)
ax = fig.add_subplot(121,projection='3d')
surf=ax.plot_surface(X, Y, rmse,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.xlabel('alpha')
plt.ylabel('lambda \n (a)')
fig.colorbar(surf)

plt.subplot(1,2,2)
plt.imshow(epart,interpolation='nearest',vmin=-1,vmax=1,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Elastic Net Penalized Partial Correlation')
plt.xlabel('\n (b)')

plt.show()