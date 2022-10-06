import numpy as np
import scipy.linalg
import random


def get_ranks_grid(X: np.array, n: int) -> np.ndarray:
    max_rank = min(X.shape)
    return np.linspace(1, max_rank, n, dtype=int, endpoint=False)


def get_known_indices(X: np.ndarray) -> list:
    return np.argwhere(~np.isnan(X)).tolist()


def split_tests_sets(known_indices: list, X: np.ndarray, split_rate: float = 0.75) -> tuple:
    test_indices = []
    train_indices = []
    X_train = np.array(X, copy=True)
    for idx in known_indices:
        ro = np.random.uniform()
        if ro < split_rate and np.count_nonzero(~np.isnan(X_train[:, idx[1]])) > 1 and np.count_nonzero(
                ~np.isnan(X_train[idx[0]])) > 1:
            test_indices.append(idx)
            X_train[tuple(idx)] = np.nan
        else:
            train_indices.append(idx)
    return test_indices, train_indices, X_train


def subspace_coherence(U: np.ndarray, r: int) -> float:
    r = np.linalg.matrix_rank(U)
    n = U.shape[0]
    subspace_coherence = -1
    for i in range(n):
        eye = np.zeros((n))
        eye[i] = 1
        # np.linalg.norm(U.reshape(3, 1) @ U.reshape(1, 3)
        PU = U @ np.linalg.inv(U.T @ U) @ U.T
        coherence = np.linalg.norm(np.dot(PU, eye)) ** 2 * (float(n) / r)
        if coherence > subspace_coherence:
            subspace_coherence = coherence
    return subspace_coherence


def matrix_coherence(X: np.ndarray) -> float:
    U, s, VT = np.linalg.svd(X, full_matrices=True, compute_uv=True)
    r = np.linalg.matrix_rank(X)
    U = U[:, :r]
    V = VT[:r, :].T
    c1 = subspace_coherence(U, r)
    c2 = subspace_coherence(V, r)
    return max(c1, c2)


def coherent_matrix() -> np.ndarray:
    u1 = (np.array([1, 0]) + np.array([0, 1])) / np.sqrt(2)
    u2 = (np.array([1, 0]) - np.array([0, 1])) / np.sqrt(2)
    u = [u1, u2]
    return sum(np.outer(ui, uj) for ui in u for uj in u)


def incoherent_subspace(n, r):
    H = scipy.linalg.hadamard(n)[:, :r]
    # H = 1 / np.sqrt(n) * scipy.linalg.hadamard(n)[:, :r]
    return H


def incoherent_matrix(n1, n2, r):
    U = incoherent_subspace(n1, r)
    V = incoherent_subspace(n2, r)
    # U, V are orthogonal
    # U, _ = np.linalg.qr(U)
    # V, _ = np.linalg.qr(V)
    output = np.zeros((n1, n2))
    for i in range(r):
        # sigma = i + 1
        sigma = random.uniform(0, 100)
        output += sigma * np.outer(U[:, i], V[:, i])

    return output


def nuclear_norm(X: np.ndarray) -> float:
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    return sum(s)


def matrix_scalar(X: np.ndarray, Y: np.ndarray) -> float:
    return np.trace(np.outer(X, Y))

def spectral_norm(X):
    sigma = np.linalg.svd(X, compute_uv=False)
    return np.max(sigma)


def PT(X: np.ndarray, PU: np.ndarray, PV: np.ndarray) -> np.ndarray:
    return PU @ X + X @ PV - PU @ X @ PV


def PT_perp(X: np.ndarray, PU: np.ndarray, PV: np.ndarray) -> np.ndarray:
    return (np.eye(*PU.shape) - PU) @ X @ (np.eye(*PV.shape) - PV)


