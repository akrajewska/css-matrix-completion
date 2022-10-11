from utils.linear_algebra import incoherent_matrix
import random
import numpy as np
import scipy


def create_rank_k_dataset(
        n_rows=5,
        n_cols=5,
        k=3,
        fraction_missing=0.7,
        symmetric=False,
        random_seed=0,
        gaussian=False,
        with_replacement=False,
        noise=0):
    # np.random.seed(random_seed)
    if gaussian:
        x = np.random.randn(n_rows, k)
        y = np.random.randn(k, n_cols)
        if noise:
            x += scipy.sparse.random(n_rows, k, density=noise)
            y += scipy.sparse.random(k, n_cols, density=noise)
        XY = np.dot(x, y)
    else:
        XY = incoherent_matrix(n_rows, n_cols, k)

    indices = [[i, j] for i in range(XY.shape[0]) for j in range(XY.shape[1])]
    if with_replacement:
        omega = random.choices(indices, k=int((1 - fraction_missing) * len(indices)))
    else:
        omega = random.sample(indices, k=int((1 - fraction_missing) * len(indices)))
    mask_array = np.zeros(XY.shape, dtype=int)

    if symmetric:
        assert n_rows == n_cols
        XY = 0.5 * XY + 0.5 * XY.T

    XY_incomplete = np.zeros(XY.shape)
    for idx in omega:
        XY_incomplete[idx[0], idx[1]] += XY[idx[0], idx[1]]
        mask_array[idx[0], idx[1]] = 1
    mask_array = mask_array.astype(bool)
    missing_mask = ~mask_array
    XY_incomplete[missing_mask] = np.nan
    return XY, XY_incomplete, omega, mask_array


def remove_pixels(
        X: np.ndarray,
        missing_square_size=32,
        random_seed=0,
        no_squares=1):
    np.random.seed(random_seed)
    height, width = X.shape
    X_missing = np.copy(X).astype('float32')
    for i in range(no_squares):
        x = np.random.randint(
            low=0,
            high=height - missing_square_size + 1)
        y = np.random.randint(
            low=0,
            high=width - missing_square_size + 1)
        X_missing[
        x: x + missing_square_size,
        y: y + missing_square_size] = np.nan
    return X_missing


def remove_pixels_uniformly(
        X: np.ndarray,
        missing_part: float = 0.9,
        random_seed=0) -> np.ndarray:
    X_missing = np.copy(X).astype('float32')
    index_nan = np.random.choice(X.size, int(missing_part * X.size), replace=False)
    X_missing.ravel()[index_nan] = np.nan
    return X_missing


def remove_pixels_with_noise(
        X: np.ndarray,
        missing_part: float = 0.6,
        random_seed=0):
    X_missing = np.copy(X).astype('float32')
    index_nan = np.random.choice(X.size, int(missing_part * X.size), replace=False)
    X_missing.ravel()[index_nan] = np.nan
    sparse_idx = np.nonzero(scipy.sparse.random(X.shape[0], X.shape[1], density=0.3))
    X_missing[sparse_idx] = np.nan
    return X_missing


def remove_random_rectangle(
        X: np.ndarray,
        r_width=32,
        r_height=0,
        random_seed=0,
        no_squares=1):
    if r_height == 0:
        r_height = X.shape[0]
    np.random.seed(random_seed)
    height, width = X.shape
    X_missing = np.copy(X).astype('float32')
    for i in range(no_squares):
        x = np.random.randint(
            low=0,
            high=height - r_height + 1)
        y = np.random.randint(
            low=0,
            high=width - r_width + 1)

        square = X_missing[
                 x: x + r_height,
                 y: y + r_width]

        square_missing = np.copy(square)
        index_nan = np.random.choice(square.size, int(0.9 * square.size), replace=False)
        square_missing.ravel()[index_nan] = np.nan
        #
        X_missing[
        x: x + r_height,
        y: y + r_width] = square_missing
    return X_missing


def remove_random_cols(
        X: np.ndarray,
        n_cols=32,
        random_seed=0,
        no_squares=1):
    X_missing = np.copy(X)
    cols_indices = np.random.choice(X.shape[1], n_cols, replace=False)
    for col in cols_indices:
        index_nan = np.random.choice(X.shape[0], int(0.9 * X.shape[0]), replace=False)
        X_missing[index_nan, col] = np.nan
    return X_missing
