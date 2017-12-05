print(__doc__)


import numpy as np
from sklearn.covariance import GraphLasso, ledoit_wolf, graph_lasso, GraphLassoCV
from sklearn.linear_model import  Lasso,ElasticNet,lasso_path,enet_path


def myglasso(data, lam=0.5):
    model=GraphLasso(alpha=lam)
    # model=GraphLassoCV()
    model.fit(data)
    cov=model.covariance_
    prec=model.precision_
    # alpha=model.alpha_
    n_samples,n_features=data.shape
    part = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            part[i, j] = -prec[i, j] / np.sqrt(prec[i, i] * prec[j, j])
    return part,prec,cov


def mylasso(data, lam):
    n_samples, n_features = data.shape
    sig=np.ones(n_features)
    tor=np.array([])
    for iter1 in range(500):
        A = np.zeros((n_features * n_samples, n_features * n_features))
        for i in range(1, n_features + 1):
            A[(i - 1) * n_samples:i * n_samples, (i - 1) * n_features:i * n_features] = data * np.tile(
                np.sqrt(sig / sig[i - 1]), (n_samples, 1))

        temp=np.eye(n_features)
        temp=temp.flatten('F')
        loc = np.array(np.nonzero(temp == 0))
        loc = loc.reshape(loc.shape[1])
        A = A[:, loc]

        y=data.flatten('F')
        model=Lasso(alpha=lam)
        model.fit(A * np.sqrt(2 * n_samples), y * np.sqrt(2 * n_samples))

        lpart=-np.eye(n_features)
        lpart[np.nonzero(lpart==0)]=model.coef_
        lpart=(lpart+lpart.T)/2

        beta = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                beta[i, j] = lpart[i, j] * np.sqrt(sig[j] / sig[i])
                beta[j, i] = lpart[j, i] * np.sqrt(sig[i] / sig[j])

        sigOld = sig
        res = data - np.dot(data, beta.T)
        sig = 1 / (res * res).mean(axis=0)
        tor = np.append(tor, np.linalg.norm(sig - sigOld))
        if tor[iter1]<1e-3:
            break
    return lpart,tor


def myelastic(data,lam,alpha):
    n_samples, n_features = data.shape
    sig = np.ones(n_features)
    tor = np.array([])
    for iter1 in range(500):
        A = np.zeros((n_features * n_samples, n_features * n_features))
        for i in range(1, n_features + 1):
            A[(i - 1) * n_samples:i * n_samples, (i - 1) * n_features:i * n_features] = data * np.tile(
                np.sqrt(sig / sig[i - 1]), (n_samples, 1))

        temp = np.eye(n_features)
        temp = temp.flatten('F')
        loc = np.array(np.nonzero(temp == 0))
        loc = loc.reshape(loc.shape[1])
        A = A[:, loc]

        y = data.flatten('F')
        model = ElasticNet(alpha=lam,l1_ratio=alpha)
        model.fit(A * np.sqrt(2*n_samples), y * np.sqrt(2*n_samples))

        epart = -np.eye(n_features)
        epart[np.nonzero(epart == 0)] = model.coef_
        epart = (epart + epart.T) / 2

        beta = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                beta[i, j] = epart[i, j] * np.sqrt(sig[j] / sig[i])
                beta[j, i] = epart[j, i] * np.sqrt(sig[i] / sig[j])

        sigOld = sig
        res = data - np.dot(data, beta.T)
        sig = 1 / (res * res).mean(axis=0)
        tor = np.append(tor, np.linalg.norm(sig - sigOld))
        if tor[iter1] < 1e-3:
            break
    return epart, tor


def mylassopath(data,lam):
    n_samples, n_features = data.shape
    sig = np.ones(n_features)
    tor = np.array([])
    for iter1 in range(500):
        A = np.zeros((n_features * n_samples, n_features * n_features))
        for i in range(1, n_features + 1):
            A[(i - 1) * n_samples:i * n_samples, (i - 1) * n_features:i * n_features] = data * np.tile(
                np.sqrt(sig / sig[i - 1]), (n_samples, 1))
        temp = np.eye(n_features)
        temp = temp.flatten('F')
        loc = np.array(np.nonzero(temp == 0))
        loc = loc.reshape(loc.shape[1])
        A = A[:, loc]
        y = data.flatten('F')
        _,coef_path,_=lasso_path(A * np.sqrt(2*n_samples), y * np.sqrt(2*n_samples),alphas=lam)
    return coef_path


def myenetpath(data,lam,alpha):
    n_samples, n_features = data.shape
    sig = np.ones(n_features)
    tor = np.array([])
    for iter1 in range(500):
        A = np.zeros((n_features * n_samples, n_features * n_features))
        for i in range(1, n_features + 1):
            A[(i - 1) * n_samples:i * n_samples, (i - 1) * n_features:i * n_features] = data * np.tile(
                np.sqrt(sig / sig[i - 1]), (n_samples, 1))
        temp = np.eye(n_features)
        temp = temp.flatten('F')
        loc = np.array(np.nonzero(temp == 0))
        loc = loc.reshape(loc.shape[1])
        A = A[:, loc]
        y = data.flatten('F')
        _, coef_path, _ = enet_path(A * np.sqrt(2 * n_samples), y * np.sqrt(2 * n_samples), alphas=lam, l1_ratio=alpha)
    return coef_path