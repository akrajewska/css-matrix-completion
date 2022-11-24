import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch

"""
Column subset selection methods
"""


def leverage_select(X, c, leverage_sort=False, random_state=None, missing_mask=None):
    """Chooses column indices from a matrix given its SVD.
    Parameters
    ----------
    V : ndarray, shape (n_features, rank)
        The set of singular vectors.
    c : float
        The expected number of columns to select.
    leverage_sort : bool
        If True, resorts the column indices in increasing order of leverage
        score. If False, the column indices are sorted normally.
    Returns
    -------
    column_indices : ndarray of ints
        An array of indices denoting which columns were selected. If
        leverage_sort was true, this array is arranged by increasing leverage
        score.
    """
    r = np.linalg.matrix_rank(X)
    tsvd = TruncatedSVD(n_components=r)
    tsvd.fit(X)
    # extract right singular vectors
    V = tsvd.components_.T[:, :r]
    # random state
    rng = np.random.RandomState(seed=None)

    # extract number of samples and rank
    n_features, k = V.shape

    # cols_missing = [column for column in range(X.shape[1]) if np.any(X[:, column] == 0)]
    # calculate normalized leverage score
    pi = np.sum(V ** 2, axis=1) / k
    # pi = np.sum(V ** 2, axis=1) / len(cols_missing)

    # iterate through columns
    column_flags = np.zeros(n_features, dtype=bool)
    for column in range(n_features):
        # for column in cols_missing:
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c * pi[column])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    column_indices = np.argwhere(column_flags).ravel()

    # if desired, sort by increasing leverage score
    if leverage_sort:
        pi_subset = pi[column_indices]
        column_indices = column_indices[np.argsort(pi_subset)]
    return column_indices


def deim_select(X, r, missing_mask=None):
    col_indices = []
    U, S, VT = np.linalg.svd(X)
    V = VT.T
    v = V[:, 0]
    p = np.argmax(np.abs(v))
    col_indices.append(p)
    for j in range(1, r):
        v = V[:, j]
        c = np.linalg.inv(V[col_indices, :j]) @ v[col_indices]
        r = v - V[:, :j] @ c
        p = np.argmax(np.abs(r))
        col_indices.append(p)
    return col_indices


def uniform(X, c, missing_mask=None, numlib='numpy'):
    if numlib:
        return np.random.choice(X.shape[1], c, replace=False)
    else:
        return torch.randperm(X.shape[1], dtype=torch.int32, device='cuda')[:c]

def freq_select(X, c, random_state=None, missing_mask=None):
    """Chooses column indices from a matrix given its SVD.
    Parameters
    ----------
    V : ndarray, shape (n_features, rank)
        The set of singular vectors.
    c : float
        The expected number of columns to select.
    leverage_sort : bool
        If True, resorts the column indices in increasing order of leverage
        score. If False, the column indices are sorted normally.
    Returns
    -------
    column_indices : ndarray of ints
        An array of indices denoting which columns were selected. If
        leverage_sort was true, this array is arranged by increasing leverage
        score.
    """

    cols_missing = [column for column in range(X.shape[1]) if np.any(X[:, column] == 0)]
    # calculate normalized leverage score
    # pi = np.sum(V ** 2, axis=1) / len(cols_missing)
    pi = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        pi[i] = len(np.argwhere(missing_mask[:, i]))
        if pi[i] == 0:
            pi[i] = 24
    pin = [p / np.sum(pi) for p in pi]
    rng = np.random.RandomState(seed=None)
    # iterate through columns
    column_flags = np.zeros(X.shape[1], dtype=bool)
    # for column in range(n_features):
    for column in range(X.shape[1]):
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c * pin[column])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    column_indices = np.argwhere(column_flags).ravel()

    return column_indices


def leverage_freq_select(X, c, random_state=None, missing_mask=None):
    alpha = 0.7
    r = np.linalg.matrix_rank(X)
    tsvd = TruncatedSVD(n_components=r)
    tsvd.fit(X)
    # extract right singular vectors
    V = tsvd.components_.T[:, :r]
    # random state
    rng = np.random.RandomState(seed=None)

    # extract number of samples and rank
    n_features, k = V.shape

    # cols_missing = [column for column in range(X.shape[1]) if np.any(X[:, column] == 0)]
    # calculate normalized leverage score
    pi1 = np.sum(V ** 2, axis=1) / k

    _pi2 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        _pi2[i] = len(np.argwhere(missing_mask[:, i]))
        if _pi2[i] == 0:
            _pi2[i] = 120
    pi2 = [p / np.sum(_pi2) for p in _pi2]
    pi = [alpha * p[0] + (1 - alpha) * p[1] for p in zip(pi1, pi2)]
    column_flags = np.zeros(X.shape[1], dtype=bool)
    # for column in range(n_features):
    for column in range(X.shape[1]):
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c * pi[column])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    column_indices = np.argwhere(column_flags).ravel()

    return column_indices


def leverage_or_freq_select(X, c, random_state=None, missing_mask=None):
    alpha = 0.5
    r = np.linalg.matrix_rank(X)
    tsvd = TruncatedSVD(n_components=r)
    tsvd.fit(X)
    # extract right singular vectors
    V = tsvd.components_.T[:, :r]
    # random state
    rng = np.random.RandomState(seed=None)

    # extract number of samples and rank
    n_features, k = V.shape

    pi1 = np.sum(V ** 2, axis=1)
    _pi = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        _pi[i] = len(np.argwhere(missing_mask[:, i]))
        if _pi[i] == 0:
            _pi[i] = pi1[i]
    pi = [p / np.sum(_pi) for p in _pi]
    column_flags = np.zeros(X.shape[1], dtype=bool)
    # for column in range(n_features):
    for column in range(X.shape[1]):
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c * pi[column])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    column_indices = np.argwhere(column_flags).ravel()

    return column_indices


def leverage_and_freq_select(X, c, random_state=None, missing_mask=None):
    alpha = 0.3
    r = np.linalg.matrix_rank(X)
    tsvd = TruncatedSVD(n_components=r)
    tsvd.fit(X)
    # extract right singular vectors
    V = tsvd.components_.T[:, :r]
    # V = tsvd.components_.T[:, :10]
    # # random state
    rng = np.random.RandomState(seed=None)

    # extract number of samples and rank
    n_features, k = V.shape
    incomplete_cols = np.argwhere([bool(len(np.argwhere(missing_mask[:, i]))) for i in range(X.shape[1])])
    complete_cols = [i for i in range(X.shape[1]) if i not in incomplete_cols]

    _pi_complete = [np.sum(V ** 2, axis=1)[i] for i in complete_cols]
    pi_complete = _pi_complete / np.sum(_pi_complete)
    c_complete = c * alpha
    c_incomplete = c * (1 - alpha)
    column_flags = np.zeros(X.shape[1], dtype=bool)
    for i, column in enumerate(complete_cols):
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c_complete * pi_complete[i])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    _pi_incomplete = np.zeros(len(incomplete_cols))
    for i in incomplete_cols:
        _pi_incomplete[i] = len(np.argwhere(missing_mask[:, i]))
    pi_incomplete = [p / np.sum(_pi_incomplete) for p in _pi_incomplete]

    for i, column in enumerate(incomplete_cols):
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c_incomplete * pi_incomplete[i])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    column_indices = np.argwhere(column_flags).ravel()

    return column_indices


def leverage_and_uniform_select(X, c, random_state=None, missing_mask=None):
    alpha = 0.4
    r = np.linalg.matrix_rank(X)
    tsvd = TruncatedSVD(n_components=r)
    tsvd.fit(X)
    # extract right singular vectors
    V = tsvd.components_.T[:, :r]
    V = tsvd.components_.T[:, :10]
    # random state
    rng = np.random.RandomState(seed=None)

    # extract number of samples and rank
    n_features, k = V.shape
    incomplete_cols = np.argwhere([bool(len(np.argwhere(missing_mask[:, i]))) for i in range(X.shape[1])])
    complete_cols = [i for i in range(X.shape[1]) if i not in incomplete_cols]

    _pi_complete = [np.sum(V ** 2, axis=1)[i] for i in complete_cols]
    pi_complete = _pi_complete / np.sum(_pi_complete)
    c_complete = c * alpha
    c_incomplete = c * (1 - alpha)
    column_flags = np.zeros(X.shape[1], dtype=bool)
    for i, column in enumerate(complete_cols):
        # Mahoney (2009), eqn 3
        # if not np.any((X[:, column] == 0)):
        #     continue
        p = min(1, c_complete * pi_complete[i])
        # selected column randomly
        column_flags[column] = p > rng.rand()

    column_flags[
        np.random.choice(incomplete_cols.ravel(), min(len(incomplete_cols), int(c_incomplete)), replace=False)] = True

    column_indices = np.argwhere(column_flags).ravel()

    return column_indices
