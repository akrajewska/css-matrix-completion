import csv
import time
import sys

sys.path.append("/home/antonina/DR/css-matrix-completion/")

from src.css_matrix_completion.css import uniform
from src.css_matrix_completion.cssmc import CSSMC, CSSMC_T
from src.css_matrix_completion.mc.soft_impute import SoftImpute, SoftImpute_T
from src.css_matrix_completion.mc.soft_impute import choose_lambda
from src.css_matrix_completion.transform import cx, cx_torch
from utils.data_generation import create_rank_k_dataset, create_rank_k_tensor

import torch

n_rows = 1000
n_cols = 1000

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


def get_errors(solution, output, missing_mask):
    return [
        torch.linalg.norm(solution - output) / torch.linalg.norm(solution),
        torch.linalg.norm(solution[missing_mask] - output[missing_mask]) / torch.linalg.norm(solution[missing_mask]),
    ]


def log(output, file_name='output'):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(output)


for trial in range(5):
    print(f'Starting trial {trial}')
    for fraction_missing in [0.5, 0.7, 0.9]:
        for rank in [5, 10]:
            print(f'Rank {rank}')
            M, M_incomplete, omega, mask_array = create_rank_k_tensor(n_rows=n_rows, n_cols=n_cols, k=rank,
                                                                       gaussian=True,
                                                                       fraction_missing=fraction_missing)

            base_log_data = [trial, n_rows, n_cols, rank, fraction_missing]
            for c_rate in [0.2, 0.5, 0.7]:
                n_selected_cols = int(c_rate * n_cols)
                solver = CSSMC_T(col_number=n_selected_cols, solver=SoftImpute_T, transform=cx_torch, col_select=uniform,
                                 fill_method='zero', max_rank=rank)
                solver.get_cols_matrix(M_incomplete, torch.isnan(M_incomplete), numlib="torch")
                best_lambda_, M_filled_old = choose_lambda(solver.C_incomplete, numlib="torch")
                start_time = time.perf_counter()
                solver.lambda_ = best_lambda_
                M_filled = solver.fit_transform(M_incomplete)
                elapsed_time = time.perf_counter() - start_time
                errors = get_errors(M_filled, M, ~mask_array)
                log_data = base_log_data + [n_selected_cols, elapsed_time] + errors
                log(log_data)

            start_time = time.perf_counter()
            # M_filled = nn_complete(M_incomplete)
            errors = get_errors(M_filled, M, ~mask_array)
            log_data = base_log_data + [n_cols, elapsed_time] + errors
            log(log_data)

    print('Finished benchmark')
