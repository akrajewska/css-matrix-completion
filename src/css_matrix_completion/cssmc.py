import numpy as np
import cvxpy as cp

from typing import Callable

from src.css_matrix_completion.css import leverage_select
from src.css_matrix_completion.transform import knn, cx, ls
from src.css_matrix_completion.mc import nn_complete


class CSSMC:

    def __init__(self, col_number: int, col_select: Callable = leverage_select, transform: Callable = knn,
                 solve: Callable = nn_complete, threshold: float = 0, fill_method='zero'):
        self.col_number = col_number
        self.col_select = col_select
        self._transform = transform
        self.threshold = threshold
        self.fill_method = fill_method
        self.solve = solve

    def fit_transform(self, X, X_correct: np.ndarray = None):
        X_tmp = np.copy(X)
        missing_mask = np.isnan(X)
        ok_mask = ~missing_mask
        self.prepare(X_tmp, missing_mask)
        if X_correct is not None:
            cols_indices = self.col_select(X_correct, missing_mask=missing_mask, c=self.col_number)
        else:
            cols_indices = self.col_select(X_tmp, missing_mask=missing_mask, c=self.col_number)
        C_incomplete = X_tmp[:, cols_indices]
        cols_missing = missing_mask[:, cols_indices]
        C_filled = self.solve(C_incomplete, cols_missing)
        if X_correct is not None:
            print(f"kurwa {np.linalg.norm(C_filled-X_correct[:, cols_indices])/np.linalg.norm(X_correct[:, cols_indices])}")
        X_filled = self.transform(X, C_filled, cols_indices, ok_mask)
        #TODO indeksy kolumn sa do debugowania
        return X_filled, cols_indices

    # def solve(self, C_incomplete, cols_ok):
    #     C = cp.Variable(C_incomplete.shape)
    #     prob = cp.Problem(cp.Minimize(cp.norm(C, p='nuc')),
    #                       [cp.multiply(cols_ok, C) == cp.multiply(cols_ok,C_incomplete)])
    #
    #     prob.solve(solver=cp.SCS, verbose=False, use_indirect=True)
    #     return C.value

    def transform(self, X_org, C_filled, cols_indices, ok_mask):
        X_filled = np.copy(X_org)
        for i, ci in enumerate(cols_indices):
            X_filled[:, ci] = C_filled[:, i]
        X_filled[ok_mask] = X_org[ok_mask]
        missing_mask = np.isnan(X_filled)
        if self._transform == cx or self._transform == ls:
            ok_mask = ~missing_mask.astype(bool)
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


    def prepare(self, X, missing_mask):
        if self.fill_method == 'zero':
            X[missing_mask] = 0
        elif self.fill_method == 'mean':
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif self.fill_method == 'median':
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif self.fill_method == 'min':
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
