import csv
import os
import time

from matplotlib import pyplot as plt
from matplotlib.image import imread
from pathlib import Path


from src.css_matrix_completion.css import uniform
from src.css_matrix_completion.cssmc import CSSMC
from src.css_matrix_completion.mc import nn_complete, svt
from src.css_matrix_completion.transform import cx
from utils.data_generation import create_rank_k_dataset, remove_pixels_uniformly

import numpy as np



def get_errors(solution, output, missing_mask):
    return [
        np.linalg.norm(solution - output) / np.linalg.norm(solution),
        np.linalg.norm(solution[missing_mask] - output[missing_mask]) / np.linalg.norm(solution[missing_mask]),
        np.mean(np.abs(solution[missing_mask] - output[missing_mask]))
    ]


def log(output, file_name='output_svt'):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(output)


path = os.path.join('DATA/PICS', 'most.pgm')
M = imread(path).astype('float32')
# plt.axis('off')
# plt.imshow(M, cmap='gray', vmin=0, vmax=400)
# plt.margins(x=0)
# plt.savefig(Path('DATA/PICS/most.jpg'),bbox_inches='tight', pad_inches=0)

n_cols = M.shape[1]
for trial in range(5):
    print(f'Starting trial {trial}')
    for missing_part in [0.9, 0.8, 0.6]:
        M_incomplete = remove_pixels_uniformly(M, missing_part=missing_part)
        missing_mask = np.isnan(M_incomplete)
        base_log_data = [trial, missing_part]
        # for c_rate in [0.5, 0.6, 0.7, 1]:
        #     n_selected_cols = int(c_rate * n_cols)
        #     solver = CSSMC(col_number=n_selected_cols, transform=cx, col_select=uniform, fill_method='zero')
        #     start_time = time.perf_counter()
        #     M_filled, cols_idx = solver.fit_transform(M_incomplete)
        #     elapsed_time = time.perf_counter() - start_time
        #     errors = get_errors(M_filled, M, missing_mask)
        #     log_data = base_log_data + [n_selected_cols, elapsed_time] + errors
        #     log(log_data)
        start_time = time.perf_counter()
        M_filled = svt(M_incomplete, np.isnan(M_incomplete))
        elapsed_time = time.perf_counter() - start_time
        errors = get_errors(M_filled, M, missing_mask)
        log_data = base_log_data + [-1, elapsed_time] + errors
        log(log_data)
    print('Finished benchmark')

