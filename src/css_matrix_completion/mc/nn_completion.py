import cvxpy as cp
import numpy as np
import fbpca


class NuclearNormMin:

    def __init__(self, max_rank=None, solver='SCS', r_svd_alg=fbpca.pca):
        self.solver = solver
        self.formulation = 'sdp'

        self.max_rank = None

    def fit_transform(self, M_incomplete, missing_mask):
        if self.formulation == 'sdp':
            return self.nn_complete(M_incomplete, missing_mask)

    def nn_complete(self, M_incomplete: np.ndarray, missing_mask: np.ndarray = None) -> np.ndarray:
        """
        Fills M_incomplete with nuclear norm minimization
        """
        if missing_mask is None:
            missing_mask = np.isnan(M_incomplete)
        M_incomplete[missing_mask] = 0
        M_filled = cp.Variable(M_incomplete.shape)
        prob = cp.Problem(cp.Minimize(cp.norm(M_filled, p='nuc')),
                          [cp.multiply(~missing_mask, M_filled) == cp.multiply(~missing_mask, M_incomplete)])

        # prob.solve(solver=cp.SCS, verbose=False, use_indirect=True)
        prob.solve(solver=cp.SCS)
        return M_filled.value
