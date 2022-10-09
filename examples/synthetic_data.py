import sys
import time

from src.css_matrix_completion.mc import nn_complete

sys.path.append("/home/antonina/DR/css-matrix-completion/")
import cvxpy as cp
from utils.data_generation import create_rank_k_dataset
from matplotlib import pyplot as plt
import numpy as np

from src.css_matrix_completion.transform import cx, ls
from src.css_matrix_completion.cssmc import CSSMC
from src.css_matrix_completion.css import uniform

n_rows = 250
n_cols = 500
n_selected_cols = 250
rank = 5
start_time = time.perf_counter()
M, M_incomplete, omega, mask_array = create_rank_k_dataset(n_rows=n_rows, n_cols=n_cols, k=rank, gaussian=True)
# missing_mask = ~mask_array.astype(bool)
# M_incomplete[missing_mask] = np.nan
missing_mask = ~mask_array
# TODO svt
# proste przyrostowe
solver = CSSMC(col_number=n_selected_cols, transform=cx, col_select=uniform, fill_method='zero')
M_filled, cols_idx = solver.fit_transform(M_incomplete, X_correct=M)
print(f"elapsed_time {time.perf_counter() - start_time}")
print(f"rmse for cscmsc {np.linalg.norm(M_filled - M) / np.linalg.norm(M)}")
print(f"rmse_omega  cscmc {np.linalg.norm(M_filled[missing_mask] - M[missing_mask]) / np.linalg.norm(M[missing_mask])}")
print(f"mae_omega cscmc {np.mean(np.abs(M_filled[missing_mask] - M[missing_mask]))}")
print(f"rmse cscmc cols {np.linalg.norm(M_filled[:, cols_idx] - M[:, cols_idx]) / np.linalg.norm(M[:, cols_idx])}")

M_filled = nn_complete(M_incomplete)

print(f"elapsed_time {time.perf_counter() - start_time}")
print(f"rmse nn {np.linalg.norm(M_filled - M) / np.linalg.norm(M)}")
print(
    f"rmse_omega nn {np.linalg.norm(M_filled[missing_mask] - M[missing_mask]) / np.linalg.norm(M[missing_mask])}")
print(f"mae_omega nn {np.mean(np.abs(M_filled[missing_mask] - M[missing_mask]))}")

cols = uniform(M, n_selected_cols)
# ok_mask = np.array(~np.isnan(M))
ok_mask = mask_array
C = np.copy(M[:, cols])
# ok_mask[:, cols] = True
X = cx(np.array(M), ok_mask.astype(bool), np.array(C))
print(f"rmse cs {np.linalg.norm(X - M) / np.linalg.norm(M)}")
