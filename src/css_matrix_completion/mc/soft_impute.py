import fbpca
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
from css_matrix_completion.errors.errors import rmse_omega, rmse
import time

def max_singular_value(M):
    _, s, _ = fbpca.pca(
        M,
        1,
        n_iter=5)
    return s[0]


def choose_lambda(M_incomplete, num_lambdas=7, numlib='numpy'):
    lib = np if numlib == 'numpy' else torch
    M_incomplete = lib.nan_to_num(M_incomplete)
    n_splits = 10
    skf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    omega = lib.argwhere(~lib.isnan(M_incomplete))
    lambda_min = 1 if numlib == 'numpy' else torch.tensor(1)
    lambda_max = SoftImpute_N._max_singular_value(
        M_incomplete) if numlib == 'numpy' else SoftImpute_T._max_singular_value(M_incomplete)
    lambda_grid = lib.exp(
        lib.linspace(lib.log(lambda_min),lib.log(lambda_max) - 1,num_lambdas))
    results = []
    M_filled_old = None
    for lambda_ in lambda_grid:
        print(f"solving for lambda_ {lambda_}")
        rmse_lambda = 0

        for train_index, test_index in skf.split(omega, None):
            M_cv = np.copy(M_incomplete) if numlib == 'numpy' else torch.clone(M_incomplete)
            for idx in omega[test_index]:
                M_cv[idx[0], idx[1]] = np.nan if numlib == 'numpy' else float('nan')
            si = SoftImpute_N(lambda_=lambda_) if numlib == 'numpy' else  SoftImpute_T(lambda_=lambda_)
            start_time = time.perf_counter()
            M_filled = si.fit_transform(M_cv, lib.isnan(M_cv), Z_init=M_filled_old)
            print(f'elapsed {time.perf_counter() - start_time}')
            rmse_lambda += rmse_omega(M_incomplete, M_filled, omega, numlib=numlib)
        M_filled_old = M_filled
        results.append(rmse_lambda / n_splits)
        print(rmse_lambda / n_splits)
    results = np.array(results) if numlib == "numpy" else torch.tensor(results)
    best_lambda_idx = lib.argmin(results)
    best_lambda = lambda_grid[best_lambda_idx]
    return best_lambda, M_filled_old


class SoftImpute:

    def __init__(self, lambda_=10, threshold=0.0001, max_iter=1000,
                 max_rank=None, numlib='numpy'):
        self.numlib = numlib
        self.lambda_ = lambda_
        self.threshold = threshold
        self.max_iter = max_iter
        self.max_rank = max_rank

    def shrinkage_operator(self, M, max_rank=None):
        print(f'start shitnking')
        (U, s, V) = self._svd(M, max_rank)
        s_shrinked = self._shrink(s)
        rank = (s_shrinked > 0).sum()
        s_shrinked = s_shrinked[:rank]
        U_shrinked = U[:, :rank]
        V_shrinked = V[:rank, :]
        S_shrinked = self._diag(s_shrinked)
        M_shrinked = U_shrinked @ (S_shrinked @ V_shrinked)
        print('shrinked')
        return M_shrinked, rank

    def solve(self, X, lambda_, Z_init=None):
        start = time.perf_counter()
        if Z_init is None:
            Z_old = self._init(X.shape)
        else:
            Z_old = Z_init
        ok_mask = self._ok_mask(X)
        for iter in range(self.max_iter):
            print('iter')
            Z_old[ok_mask] = X[ok_mask]
            Z_new, rank = self.shrinkage_operator(Z_old)
            if self._converged(Z_new, Z_old):
                print(f"Converged after {iter} iterations and rankd {rank} ")
                break
            Z_old = Z_new
        Z_new[ok_mask] = X[ok_mask]
        print(f'Solved in {time.perf_counter()-start}')
        return Z_new

    def _converged(self, Z_new, Z_old):
        return self._rmse(Z_old, Z_new) < self.threshold

    def fit(self, M, missing_mask, Z_init=None):
        return self.solve(M, self.lambda_, Z_init=Z_init)

    def transform(self, M, best_lambda_, Z_new):
        return self.solve(M, best_lambda_, Z_init=Z_new)

    def fit_transform(self, M, missing_mask, Z_init=None):
        return self.fit(M, missing_mask, Z_init=Z_init)


class SoftImpute_N(SoftImpute):

    @classmethod
    def _max_singular_value(cls, M):
        _, s, _ = fbpca.pca(
            M,
            1,
            n_iter=5)
        return s[0]

    def _svd(self, M, max_rank=None):
        if max_rank:
            (U, s, V) = fbpca.pca(
                M,
                max_rank)

        else:
            (U, s, V) = np.linalg.svd(
                M,
                full_matrices=False,
                compute_uv=True)
        return (U, s, V)

    def _shrink(self, s):
        """
        Shrinks singular vector with self.lambda_
        """
        return np.maximum(s - self.lambda_, 0)

    def _diag(self, s):
        return np.diag(s)

    def _init(self, shape):
        return np.zeros(shape)

    def _ok_mask(self, X):
        return ~np.isnan(X)

    @classmethod
    def _omega(cls, X):
        return np.argwhere(~np.isnan(X))

    def _rmse(self, X, Y):
        return np.linalg.norm(X - Y) / np.linalg.norm(X)


class SoftImpute_T(SoftImpute):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    @classmethod
    def _max_singular_value(cls, M):
        _, s, _ = torch.svd_lowrank(
            M,
            1)
        return s[0]

    def _svd(self, M, max_rank=None):
        if max_rank:
            (U, s, V) = torch.svd_lowrank(
                M,
                max_rank)

        else:
            (U, s, V) = torch.linalg.svd(
                M,
                full_matrices=False)
        return (U, s, V)

    def _shrink(self, s):
        """
        Shrinks singular vector
        """
        return torch.maximum(s - self.lambda_, torch.tensor(0))

    def _diag(self, s):
        return torch.diag(s)

    def _init(self, shape):
        return torch.zeros(shape, dtype=torch.float64, device=self.device)

    def _ok_mask(self, X):
        return ~torch.isnan(X)

    @classmethod
    def _omega(cls, X):
        return torch.argwhere(~torch.isnan(X))

    def _rmse(self, X, Y):
        return torch.linalg.norm(X - Y) / torch.linalg.norm(X)

    @classmethod
    def _lambda_grid(cls, X, num_lambdas=7):
        lambda_grid = torch.exp(
            torch.linspace(start=torch.log(1), stop=torch.log(max_singular_value(X)) - 1, num=num_lambdas))
        return lambda_grid
