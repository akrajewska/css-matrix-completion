import cvxpy as cp
import numpy as np

"""
Matrix completion methods
"""


def nn_complete(M_incomplete: np.ndarray) -> np.ndarray:
    """
    Fills M_incomplete with nuclear norm minimization
    """
    missing_mask = np.isnan(M_incomplete)
    M_incomplete[missing_mask] = 0
    M_filled = cp.Variable(M_incomplete.shape)
    prob = cp.Problem(cp.Minimize(cp.norm(M_filled, p='nuc')),
                      [cp.multiply(~missing_mask, M_filled) == cp.multiply(~missing_mask, M_incomplete)])

    prob.solve(solver=cp.SCS, verbose=False)
    return M_filled.value
