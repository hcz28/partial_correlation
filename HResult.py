print(__doc__)
# Compare the estiamted partial correlation with the ground truth
# Figure 7

from MyFunc import myglasso,mylasso,myelastic,myenetpath,mylassopath
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_sparse_spd_matrix

# load data
X=np.genfromtxt('data/data.csv',delimiter=',')
cov=np.genfromtxt('data/true_cov.csv',delimiter=',')
prec=np.genfromtxt('data/true_prec.csv',delimiter=',')

n_samples, n_features=X.shape
part=np.zeros((n_features,n_features))
for i in range(n_features):
    for j in range(n_features):
        part[i,j]=-prec[i,j]/np.sqrt(prec[i,i]*prec[j,j])

# Graphical lasso
glam=np.array([0.4141])
gpart,gprec,gcov=myglasso(X, lam=glam)
grmse=np.linalg.norm(gpart - part, ord='fro')

# Lasso
llam=np.array([0.5922])
lpart,tor=mylasso(X,lam=llam)
lrmse=np.linalg.norm(lpart - part, ord='fro')

# Elastic Net
elam=np.array([1.3368])
ealpha=np.array([0.2678])
epart, tor=myelastic(X, lam=elam,alpha=ealpha)
ermse=np.linalg.norm(epart - part, ord='fro')

print('graphical lasso error: %f\nlasso error: %f\nelastic net error: %f' %(grmse,lrmse,ermse))

# plot
plt.figure(1)
plt.subplots_adjust(left=0.02,right=0.98)
vmax=np.maximum(part.max(),abs(part.min()))
plt.subplot(1,4,1)
plt.imshow(gpart, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Graphical Lasso, RSE=%f' % grmse)
plt.subplot(1,4,2)
plt.imshow(lpart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Lasso Penalized, RSE=%f' % lrmse)
plt.subplot(1,4,3)
plt.imshow(epart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Elastic Net Penalized, RSE=%f' % ermse)
plt.subplot(1,4,4)
plt.imshow(part,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('True Partial Correlation')
plt.show()

plt.figure(2)
t=np.triu(np.ones((n_features,n_features)),1)
part2=part[np.nonzero(t)]
gpart2=gpart[np.nonzero(t)]
lpart2=lpart[np.nonzero(t)]
epart2=epart[np.nonzero(t)]
plt.subplot(3,1,1)
plt.plot(part2.flatten(),color='r',linestyle='--',label='True')
plt.plot(gpart2.flatten(),color='b',label='Graphical lasso')
plt.legend(prop={'size':10})
plt.xlabel('locations')
plt.ylabel('coefficient')

plt.subplot(3,1,2)
plt.plot(part2.flatten(),color='r',linestyle='--',label='True')
plt.plot(lpart2.flatten(),label='Lasso')
plt.legend(prop={'size':10})
plt.xlabel('locations')
plt.ylabel('coefficient')

plt.subplot(3,1,3)
plt.plot(part2.flatten(),color='r',linestyle='--',label='True')
plt.plot(epart2.flatten(),label='Elastic Net')
plt.legend(prop={'size':10})
plt.xlabel('locations')
plt.ylabel('coefficient')

plt.show()