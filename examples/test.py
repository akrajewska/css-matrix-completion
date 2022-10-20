import dask_ml
import numpy as np
import dask
import dask.array as da
from css_matrix_completion.css import uniform
from utils.data_generation import create_rank_k_dataset


def grad_solve_dask(X, y):
    _, s, _ = np.linalg.svd(2 * X.T.dot(X))
    step_size = 1 / s - 1e-8
    #step_size = 0.01
    ## define some parameters
    max_steps = 1000
    tol = 1e-8
    beta_hat = np.zeros(X.shape[1])
    for k in range(max_steps):
        if k == 18:
            print("test 18")
        if k == 17:
            print("test 17")
        Xbeta = X.dot(beta_hat)
        func = ((y - Xbeta)**2).sum()
        gradient = 2 * X.T.dot(Xbeta - y)

        ## Update

        obeta = beta_hat
        beta_hat = beta_hat - step_size * gradient
        new_func = ((y - X.dot(beta_hat))**2).sum()

        if np.any(np.isinf(beta_hat)):
            print(f"k {k}")
            print(f"beta_hat {beta_hat}")
        beta_hat, func, new_func = dask.compute(beta_hat, func, new_func)  # <--- Dask code
        if np.any(np.isinf(beta_hat)):
            print(f"k after {k}")
            print(f"beta_hat {beta_hat}")
        ## Check for convergence
        change = np.absolute(beta_hat - obeta).max()

        if change < tol:

            break
    print(f"Finished after iteration {k}")
    return beta_hat


# beta = np.random.random(250)  # random beta coefficients, no intercept
# X = dask.array.random.normal(0, 1, size=(1000, 250), chunks=(1000, 250))
# y = X.dot(beta) + dask.array.random.normal(0, 1, size=1000, chunks=(1000,))
# print(y.shape)

# create inputs with a bunch of independent normals
beta = np.random.random(100)  # random beta coefficients, no intercept
X = da.random.normal(0, 1, size=(1000000, 100), chunks=(100000, 100))
y = X.dot(beta) + da.random.normal(0, 1, size=1000000, chunks=(100000,))

X, y = dask.persist(X, y)
sol = grad_solve_dask(X, y)

# M, M_incomplete, omega, mask_array = create_rank_k_dataset(100, 100, 10, gaussian=True)
# cols_selected = uniform(M, 25)
# C = M[:, cols_selected]
# sol = grad_solve_dask(C, M[:,1])
print('finito')
print(sol.shape)
sol1 = np.linalg.lstsq(X,y)[0]
print('sol1.shape')
print(sol1.shape)
sol = dask.compute(sol)
print(dask.compute(np.linalg.norm(sol-sol1)))
print(sol)