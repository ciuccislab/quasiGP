#__authors__ = 'Francesco Ciucci, Adeleke Maradesa'

#import necessary library
from numpy import linalg as la
import numpy as np

# find the nearest positive-definite matrix

def is_PD(A): 
      
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_PD(A):
    
    # based on N.J. Higham (1988) https://doi.org/10.1016/0024-3795(88)90223-6 (https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd)

    B = (A + A.T)/2
    _, Sigma_mat, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm):
        eps = np.spacing(la.norm(A_symm))
        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm
