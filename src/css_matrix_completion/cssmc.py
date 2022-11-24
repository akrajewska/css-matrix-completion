from typing import Callable

import numpy as np
import torch

from css_matrix_completion.mc.soft_impute import SoftImpute
from src.css_matrix_completion.css import leverage_select
from src.css_matrix_completion.mc.nn_completion import NuclearNormMin
from src.css_matrix_completion.transform import knn, cx, ls


class CSSMC:

    def __init__(self, col_number: int, col_select: Callable = leverage_select, transform: Callable = knn,
                 solver: Callable = NuclearNormMin, threshold: float = 0, fill_method='zero', lambda_=0, max_rank=None):
        self.col_number = col_number
        self.col_select = col_select
        self._transform = transform
        self.threshold = threshold
        self.fill_method = fill_method
        self.solver = solver
        self.lambda_ = lambda_
        self.max_rank = max_rank
        self.C_incomplete = None

    def get_cols_matrix(self, X, missing_mask, numlib='numpy'):
        self.cols_indices = self.col_select(X, missing_mask=missing_mask, c=self.col_number, numlib=numlib)
        self.C_incomplete = self._copy(X[:, self.cols_indices])
        self.cols_missing = missing_mask[:, self.cols_indices]

    def fit_transform(self, X):
        X_tmp = self._copy(X)
        missing_mask = self._missing_mask(X)
        ok_mask = ~missing_mask
        self._prepare(X_tmp, missing_mask)
        if self.C_incomplete is None:
            cols_indices = self.col_select(X_tmp, missing_mask=missing_mask, c=self.col_number, numlib=self.numlib)
            C_incomplete = X_tmp[:, cols_indices]
            cols_missing = missing_mask[:, cols_indices]
        else:
            cols_indices = self.cols_indices
            C_incomplete = self.C_incomplete
            cols_missing = self.cols_missing

        C_filled = self.fill_columns(C_incomplete, cols_missing)
        X_filled = self.transform(X, C_filled, cols_indices, ok_mask)
        return X_filled

    def fill_columns(self, C_incomplete, cols_missing):
        if self.lambda_:
            solver = self.solver(lambda_=self.lambda_, max_rank=self.max_rank)
        else:
            solver = self.solver()
        return solver.fit_transform(C_incomplete, cols_missing)

    def transform(self, X_org, C_filled, cols_indices, ok_mask):
        X_filled = self._copy(X_org)
        for i, ci in enumerate(cols_indices):
            X_filled[:, ci] = C_filled[:, i]
        X_filled[ok_mask] = X_org[ok_mask]
        missing_mask = self._missing_mask(X_filled)
        if self._transform == cx or self._transform == ls:
            ok_mask = ~missing_mask
            return self._transform(X_filled, ok_mask, C_filled)
        return self._transform(X_filled, ~missing_mask)

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values



class CSSMC_N(CSSMC):

    numlib = 'numpy'

    def _copy(self, X):
        return np.copy(X)

    def _missing_mask(self, X):
        return np.isnan(X)

    def _prepare(self, X, missing_mask):
        if self.fill_method == 'zero':
            X[missing_mask] = 0
        elif self.fill_method == 'mean':
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif self.fill_method == 'median':
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif self.fill_method == 'min':
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)

class CSSMC_T(CSSMC):
    numlib = 'torch'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    def _copy(self, X):
        return torch.clone(X)

    def _missing_mask(self, X):
        return torch.isnan(X)

    def _prepare(self, X, missing_mask):
        X[missing_mask] = 0

