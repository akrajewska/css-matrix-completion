import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
def rmse(X, Y):
    return np.linalg.norm(X-Y)/np.linalg.norm(X)

def rmse_omega(X, Y, omega, numlib = 'numpy'):
    diff = ((X[i[0], i[1]] - Y[i[0], i[1]])**2 for i in omega)
    x = ((X[i[0], i[1]])**2 for i in omega)
    if numlib == 'numpy':
        return np.sum(diff)/np.sum(x)
    return torch.sum(torch.tensor(list(diff), dtype=torch.float64, device=device))/torch.sum(torch.tensor(list(x), dtype=torch.float64, device=device))
