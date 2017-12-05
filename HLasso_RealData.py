# print(__doc__)
# Tuning the lasso parameter and show the best estimation
# Figure 4,5

from MyFunc import mylasso, myglasso

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# path1 = 'D:\Other\FYP ZJU\DATA&CODE_WD\BC_network _new.mat'
# BC = sio.loadmat(path1)
# X = BC['BC3'].T          # pxn matrix to nxp matrix
# X -= X.mean(axis=0)
# X /= X.std(axis=0)


mainDir = 'D:\\Program Files\\MATLAB\\Brain network\\Signal\\'
group = 'RA'
num = 6
X=np.genfromtxt(mainDir+group+'_'+str(num)+'.csv',delimiter=',')
X=X.astype('float64')
myScaler=preprocessing.StandardScaler()
X=myScaler.fit_transform(X)
n_samples,n_features=X.shape
# Lasso
# lam=np.linspace(0.32,0.9) #0.25-1
lam=np.array([0.3])
# rmse=np.array([])
for i in range(lam.shape[0]):
    lpart,_,_=myglasso(X,lam=lam[i])
    # rmse = np.append(rmse, np.linalg.norm(lpart-part,ord='fro'))

# plot result
plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.05,right=0.98)
vmax=1

# plt.subplot(1,2,1)
plt.imshow(lpart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Lasso Penalized Partial Correlation')
plt.xlabel('\n (b)')

# plt.subplot(1,2,2)
# plt.title('lambda='+str(lam))
# plt.plot(tor,'bo-')
# plt.xlabel('iterations\n (b)')
# plt.ylabel('tor')
plt.show()
