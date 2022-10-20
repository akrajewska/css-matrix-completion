from typing import Optional

import cvxpy as cp
import numpy as np

from fancyimpute import SoftImputeWarmStarts
"""
Matrix completion methods
"""


def nn_complete(M_incomplete: np.ndarray, missing_mask: np.ndarray=None) -> np.ndarray:
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


def svt(M_incomplete, missing_mask):
    solver = SoftImputeWarmStarts(with_iteration_number=True)
    M_incomplete[missing_mask] = np.nan
    ret = solver.fit_transform(M_incomplete)
    _, M_filled, _ = ret[-1]
    return M_filled


# def svt(M_incomplete: np.ndarray, t: Optional[float] = None) -> np.ndarray:
#     """
#     Naive implemenatation of SVT algorithm
#     """
#     M_tmp = np.copy(M_incomplete)
#     missing_mask = np.isnan(M_tmp)
#     M_tmp[missing_mask] = 0
#     U, S, VT = np.linalg.svd(M_tmp)
#     if t is None:
#         t = np.max(S) - 50
#     Sigma = np.zeros(M_tmp.shape)
#     S = [max(s - t, 0) for s in S]
#     np.fill_diagonal(Sigma, S)
#     return U @ Sigma @ VT


def grid_svt(M_incomplete: np.ndarray, k: int = 5) -> np.ndarray:
    """
    SVT with warm restarts
    """
    M_tmp = np.copy(M_incomplete)
    missing_mask = np.isnan(M_tmp)
    ok_mask = ~missing_mask
    M_tmp[missing_mask] = 0
    U, S, VT = np.linalg.svd(M_tmp)
    t_max = np.max(S)
    t_min = 0
    M_t = M_tmp
    for t in np.linspace(t_max, t_min, num=k):
        for i in range(1000):
            M_t = svt(M_t, t)
            M_t[ok_mask] = M_incomplete[ok_mask]
    return M_t
