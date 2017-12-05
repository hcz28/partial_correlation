print(__doc__)
#

from MyFunc import myglasso, mylasso
import numpy as np
from sklearn import covariance, preprocessing
import matplotlib.pyplot as plt

group='RA/'
numSubj=31
for subj in np.arange(1,numSubj+1):
    # load data
    Dir='D:/Program Files/MATLAB/Brain network/Signal/'+group
    X=np.genfromtxt(Dir+str(subj)+'.csv',delimiter=',')
    X=X.astype('float64')
    myScaler=preprocessing.StandardScaler()
    X=myScaler.fit_transform(X)
    n_samples,n_features=X.shape

    # gpart=np.zeros([80,80])
    # gam=np.linspace(0.01, 1)
    try:
        gpart,gprec,gcov,alpha=myglasso(X)
        print('The graphical lasso parameter is %f' % alpha)
        np.savetxt(Dir + 'PC' + str(subj) + '.csv', gpart, delimiter=",")
    except FloatingPointError:
        print('error')


    # lam=np.array([0.2])
    # for i in range(lam.shape[0]):
    #     lpart,tor=mylasso(X,lam=lam[i])

    # plot
    # plt.figure(figsize=(8,6))
    # vmax=1
    #
    # # plt.subplots_adjust(left=0.05,right=0.98)
    # # plt.subplot(1,2,1)
    # # plt.imshow(lpart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
    # # plt.colorbar()
    # # plt.xlabel('\n (a)')
    # # plt.title('Partial Correlation')
    #
    # # plt.subplot(1,2,2)
    # plt.imshow(gpart,interpolation='nearest',vmin=-vmax,vmax=vmax,cmap=plt.cm.RdBu_r)
    # plt.colorbar()
    # plt.xlabel('\n (b)')
    # plt.title('Graphical Lasso')
    # plt.show()