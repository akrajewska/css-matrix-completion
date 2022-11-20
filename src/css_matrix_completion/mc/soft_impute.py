import fbpca
import numpy as np
from sklearn.model_selection import train_test_split

from css_matrix_completion.errors.errors import rmse_omega, rmse


class SoftImputeWarmStarts:
    def __init__(self, threshold=0.0001, r_svd_alg=fbpca.pca, number_of_lambdas=150, max_iter=1000, max_rank=None):
        self.number_of_lambdas = number_of_lambdas
        self.with_warm_starts = False

    def fit(self, M, missing_mask):
        M_train, omega_test = self._get_M_train(M, missing_mask)
        solution_error = np.infty
        for lambda_ in self._lambda_grid(M):
            Z_new = self.solve(M_train, lambda_)
            if new_error := self._test_error(M, Z_new, omega_test) < solution_error:
                print(f"Updating for lambda {lambda_}")
                best_lambda_ = lambda_
                solution_error = new_error
        return best_lambda_, Z_new


class SoftImpute:

    def __init__(self, lambda_, threshold=0.0001, r_svd_alg=fbpca.pca, number_of_lambdas=150, max_iter=1000,
                 max_rank=None):
        self.lambda_ = lambda_
        self.threshold = threshold
        self.r_svd_alg = r_svd_alg
        self.max_rank = None
        self.max_iter = max_iter

    def _lambda_grid(self, M):
        return np.exp(
            np.linspace(start=np.log(self._max_singular_value(M)), stop=np.log(1), num=150))

    def _max_singular_value(self, M):
        _, s, _ = fbpca.pca(
            M,
            1,
            n_iter=5)
        return s[0]

    def _get_M_train(self, M, missing_mask):
        omega = np.argwhere(~missing_mask)
        omega_train, omega_test = train_test_split(omega, test_size=0.33)
        M_train = np.empty(M.shape)
        M_train[:] = np.nan
        for idx in omega_train:
            M_train[tuple(idx)] = M[tuple(idx)]
        return M_train, omega_test

    def _test_error(self, M, Z, omega_test):
        return rmse_omega(M, Z, omega_test)

    def _converged(self, Z_new, Z_old):
        rmse_ = rmse(Z_old, Z_new)
        # print(f"RMSE {rmse_}")
        return rmse_ < self.threshold

    def shrinkage_operator(self, M, lambda_, max_rank=None):
        if max_rank:
            (U, s, V) = self.r_svd_alg(
                M,
                max_rank)

        else:
            (U, s, V) = np.linalg.svd(
                M,
                full_matrices=False,
                compute_uv=True)
        s_shrinked = np.maximum(s - lambda_, 0)
        rank = (s_shrinked > 0).sum()
        s_shrinked = s_shrinked[:rank]
        U_shrinked = U[:, :rank]
        V_shrinked = V[:rank, :]
        S_shrinked = np.diag(s_shrinked)
        M_filled = np.dot(U_shrinked, np.dot(S_shrinked, V_shrinked))
        # print(f"Matrix rank {rank}")
        return M_filled, rank

    def solve(self, M, lambda_, Z_init=None):
        if Z_init is None:
            Z_old = np.zeros(M.shape)
        else:
            Z_old = Z_init
        missing_mask = np.isnan(M)
        ok_mask = ~np.isnan(M)
        for iter in range(self.max_iter):
            Z_old[ok_mask] = M[ok_mask]
            Z_new, rank = self.shrinkage_operator(Z_old, lambda_)
            if self._converged(Z_new, Z_old):
                print(f"Converged after {iter} iterations and rankd {rank} ")
                break
            Z_old = Z_new
        Z_new[ok_mask] = M[ok_mask]
        return Z_new

    def fit(self, M, missing_mask, Z_init=None):
        return self.solve(M, self.lambda_, Z_init=Z_init)

    def transform(self, M, best_lambda_, Z_new):
        return self.solve(M, best_lambda_, Z_init=Z_new)

    def fit_transform(self, M, missing_mask, Z_init=None):
        # czy oplaca sie robić warmstart z Z_new
        # best_lambda_, Z_new = self.fit(M, missing_mask)
        return self.fit(M, missing_mask, Z_init=Z_init)
