import csv
import time
import sys
sys.path.append("/home/antonina/DR/css-matrix-completion/")

from src.css_matrix_completion.css import uniform
from src.css_matrix_completion.cssmc import CSSMC
from src.css_matrix_completion.mc import nn_complete
from src.css_matrix_completion.transform import cx
from utils.data_generation import create_rank_k_dataset

import numpy as np

n_rows = 100
n_cols = 500


def get_errors(solution, output, missing_mask):
    return [
        np.linalg.norm(solution - output) / np.linalg.norm(solution),
        np.linalg.norm(solution[missing_mask] - output[missing_mask]) / np.linalg.norm(solution[missing_mask]),
        np.mean(np.abs(solution[missing_mask] - output[missing_mask]))
    ]


def log(output, file_name='output'):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(output)


for trial in range(1):
    print(f'Starting trial {trial}')
    for noise in [0, 0.2, 0.5]:
        for rank in [5, 10, 15]:
            print(f'Rank {rank}')
            M, M_incomplete, omega, mask_array = create_rank_k_dataset(n_rows=n_rows, n_cols=n_cols, k=rank,
                                                                       gaussian=True,
                                                                       noise=noise)
            base_log_data = [trial, n_rows, n_cols, rank]
            for c_rate in [0.2, 0.5, 0.7]:
                n_selected_cols = int(c_rate * n_cols)
                solver = CSSMC(col_number=n_selected_cols, transform=cx, col_select=uniform, fill_method='zero')
                start_time = time.perf_counter()
                M_filled, cols_idx = solver.fit_transform(M_incomplete)
                elapsed_time = time.perf_counter() - start_time
                errors = get_errors(M_filled, M, ~mask_array)
                log_data = base_log_data + [n_selected_cols, elapsed_time] + errors
                log(log_data)

            start_time = time.perf_counter()
            M_filled = nn_complete(M_incomplete)
            errors = get_errors(M_filled, M, ~mask_array)
            log_data = base_log_data + [n_cols, elapsed_time] + errors
            log(log_data)

    print('Finished benchmark')

