import fbpca
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from utils.data_generation import create_rank_k_dataset
from src.css_matrix_completion.mc.soft_impute import SoftImpute
from fancyimpute import SoftImpute as SoftImputeF
from css_matrix_completion.errors.errors import rmse_omega, rmse
import pandas as pd

n_splits = 10
skf = KFold(n_splits=n_splits, random_state=None, shuffle=True)

M, M_incomplete, omega, mask_array = create_rank_k_dataset(n_rows=100, n_cols=100, k=5,
                                                           gaussian=True,
                                                           fraction_missing=0.8)
def max_singular_value(M_incomplete):
    M_incomplete_tmp = np.copy(M_incomplete)
    M_incomplete_tmp[np.isnan(M_incomplete)] = 0
    _, s, _ = fbpca.pca(
        M_incomplete_tmp,
        1,
        n_iter=5)
    return s[0]


omega = np.array(omega)



# best_lambda, M_filled_old = best_lambda(M_incomplete)
# si = SoftImpute(lambda_=best_lambda)
# M_filled = si.fit_transform(M_incomplete, np.isnan(M_incomplete), Z_init=M_filled_old)
# print(f"rmse fineal {rmse(M, M_filled)}")