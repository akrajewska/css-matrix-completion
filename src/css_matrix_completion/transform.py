import numpy as np
from fancyimpute import KNN, IterativeImputer, IterativeSVD
import scipy


def svt(X, t=None):
    U, S, VT = np.linalg.svd(X)
    if t is None:
        t = np.max(S) - 50
    Sigma = np.zeros(X.shape)
    S = [max(s - t, 0) for s in S]
    np.fill_diagonal(Sigma, S)
    return U @ Sigma @ VT


def grid_svt(X, ok_mask, k=5):
    U, S, VT = np.linalg.svd(X)
    t_max = np.max(S)
    t_min = 0
    X_t = X
    for t in np.linspace(t_max, t_min, num=k):
        X_t = svt(X_t, t)
        X_t[ok_mask] = X[ok_mask]
    return X_t


def knn(X, ok_mask):
    solver = KNN()
    missing_mask = ~ok_mask
    X_incomplete = X.copy()
    X_out = solver.fit_transform(X_incomplete, missing_mask)
    return X_out


def mice(X, ok_mask):
    solver = IterativeImputer(max_iter=1, verbose=True)
    missing_mask = ~ok_mask
    X_incomplete = X.copy()
    X_out = solver.fit_transform(X_incomplete, missing_mask)
    return X_out


def svd(X, r):
    U, S, VT = np.linalg.svd(X)
    Sigma = np.zeros(X.shape)
    S = [max(s) for s in S[:r]]
    np.fill_diagonal(Sigma, S)
    return U @ Sigma @ VT


def iterative_svd(X, ok_mask, r=10):
    solver = IterativeSVD(max_iters=1, rank=10)
    missing_mask = ~ok_mask
    return solver.fit_transform(X, missing_mask)


def ls(X, ok_mask):
    diag = ok_mask.T.flatten()
    # return np.diag(diag)
    A = np.diagflat(diag)
    b = np.multiply(ok_mask, X).T.flatten()
    b[np.argwhere(np.isnan(b))] = 0
    x, *_ = scipy.linalg.lstsq(A, b, lapack_driver='gelsy', check_finite=False)
    return x.reshape((X.shape[1], X.shape[0])).T


def cx(X, ok_mask, C):
    m, n = X.shape
    _, k = C.shape
    Y = np.zeros((n, k))
    # For each row, solve a k-dimensional regression problem
    # only over the nonzero projection entries. Note that the
    # projection changes the least-squares matrix siX so we
    # cannot vectorize the outer loop.
    ok_mask = ok_mask.astype(bool)
    for i in range(n):
        si = ok_mask[:, i]
        sia = X[si, i]
        siX = C[si]
        Y[i, :] = np.linalg.lstsq(siX, sia)[0]
    return C @ Y.T
    # YT = np.multiply(ok_mask, X)@C
    # return YT.T


def update_right(A, S, X):
    """Update right factor for matrix completion objective."""
    m, n = A.shape
    _, k = X.shape
    Y = np.zeros((n, k))
    # For each row, solve a k-dimensional regression problem
    # only over the nonzero projection entries. Note that the
    # projection changes the least-squares matrix siX so we
    # cannot vectorize the outer loop.
    for i in range(n):
        si = S[:, i]
        sia = A[si, i]
        siX = X[si]
        Y[i, :] = np.linalg.lstsq(siX, sia)[0]
    return Y
