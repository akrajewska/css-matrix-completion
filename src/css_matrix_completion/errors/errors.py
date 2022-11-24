import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
def rmse(X, Y):
    return np.linalg.norm(X-Y)/np.linalg.norm(X)

def rmse_omega(X, Y, ok_mask, numlib = 'numpy'):
    lib = np if numlib == 'numpy' else torch
    return lib.norm(X[ok_mask] - Y[ok_mask])/lib.norm(X[ok_mask])

def rmse_torch(X, Y, ok_mask):
    return torch.norm(X[ok_mask] - Y[ok_mask])/torch.norm(X[ok_mask])