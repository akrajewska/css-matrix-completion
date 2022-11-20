import numpy as np


def rmse(X, Y):
    return np.linalg.norm(X-Y)/np.linalg.norm(X)

def rmse_omega(X, Y, omega):
    return np.sum((X[i[0], i[1]] - Y[i[0], i[1]])**2 for i in omega)/np.sum((X[i[0], i[1]])**2 for i in omega)