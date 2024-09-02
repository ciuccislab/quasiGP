# import necessary package
from numpy.linalg import cholesky, solve
import numpy as np
import nearest_PD as nPD
from numpy.random import randn
from numpy.matlib import repmat


def generate_tmg(F, g, M, mu_r, initial_X, cov=True, L=1):
    """
    Generate samples from a d-dimensional Gaussian (with constraints given by F*X+g >0) using Hamiltonian Monte Carlo (HMC) with an adaptive step size strategy.
    
    This algorithm follows HMC by Ari Pakman, as described in http://arxiv.org/abs/1208.4118
    
    Inputs:
    - F: constraint matrix
    - g: constraint vector
    - M: covariance or precision matrix
    - mu_r: mean vector
    - initial_X: vector of initial sample
    - cov: boolean indicating whether M is a covariance matrix (True) or a precision matrix (False)
    - L: number of samples to generate
    
    Output:
    - Xs: generated samples from a d-dimensional Gaussian under the constraints that F*X+g >0 
    """
    
    # Number of constraints
    m = g.shape[0]
    
    # Dimension of the problem
    d = initial_X.shape[0]
    bounce_count = 0
    near_zero = 1E-12

    # Check if the initial condition is consistent
    if np.any(F @ initial_X + g < 0):
        print("Error: Inconsistent initial condition")
        return
    
    # Adjust matrices and vectors based on whether M is a covariance matrix or precision matrix
    if cov:
        # Ensure M is positive definite
        M = nPD.nearest_PD(M) if not nPD.is_PD(M) else M
        mu = mu_r
        g = g + F @ mu
        R = cholesky(M).T  # Use upper triangular matrix
        F = F @ R.T
        initial_X = initial_X - mu
        initial_X = solve(R.T, initial_X)
    else:
        # Ensure M is positive definite
        M = nPD.nearest_PD(M) if not nPD.is_PD(M) else M
        r = mu_r
        R = cholesky(M).T  # Use upper triangular matrix
        mu = solve(R, solve(R.T, r))
        g = g + F @ mu
        F = solve(R, F)
        initial_X = initial_X - mu
        initial_X = R @ initial_X
    
    # squared Euclidean norm of constraint matrix columns
    F2 = np.sum(F ** 2, axis=1)
    Ft = F.T

    # Initialize storage for samples
    Xs = np.zeros((d, L))
    Xs[:, 0] = initial_X
    
    # Parameters for adaptive HMC
    epsilon = 1.8  # Initial step size
    adapt_interval = 50  # Interval for adapting the step size
    adapt_factor = 0.01  # Factor for adjusting the step size
    target_acceptance_rate = 0.90  # Desired acceptance rate for samples
    
    last_X = initial_X
    i = 2
    adapt_count = 0

    while i <= L:
        if i % 1000 == 0:
            print('Current sample number', i, '/', L)

        stop = False
        j = -1
        # generate inital velocity from normal distribution
        V0 = np.random.randn(d)

        X = last_X
        T = np.pi / 2
        tt = 0

        while True:
            a = np.real(V0)
            b = X

            fa = F @ a
            fb = F @ b

            U = np.sqrt(fa ** 2 + fb ** 2)
            phi = np.arctan2(-fa, fb)
            
            # find the locations where the constraints were hit
            pn = np.abs(g / U) <= 1

            if pn.any():
                inds = np.where(pn)[0]
                phn = phi[pn]
                t1 = -phn + np.arccos(-g[pn] / U[pn])
            
            
            # if there was a previous reflection (j > -1)
            # and there is a potential reflection at the sample plane
            # make sure that a new reflection at j is not found because of numerical error
                if j > -1:
                    if pn[j]:
                        indj = np.cumsum(pn)[j] - 1
                        tt1 = t1[indj]
                        if np.abs(tt1) < near_zero or np.abs(tt1 - np.pi) < near_zero:
                            t1[indj] = np.inf

                mt = np.min(t1)
                m_ind = np.argmin(t1)
                j = inds[m_ind]
            else:
                mt = T
            
            # Update travel time
            tt = tt + mt

            if tt >= T:
                mt = mt - (tt - T)
                stop = True
            
            # Update position and velocity
            X = a * np.sin(mt) + b * np.cos(mt)
            V = a * np.cos(mt) - b * np.sin(mt)

            if stop:
                break
            
            # Update new velocity
            qj = F[j, :] @ V / F2[j]
            V0 = V - 2 * qj * Ft[:, j]
            bounce_count += 1

        if np.all(F @ X + g > 0):
            Xs[:, i - 1] = X
            last_X = X
            i += 1
            # Adaptive step size strategy
            if i % adapt_interval == 0:
                acceptance_rate = adapt_count / adapt_interval
                if acceptance_rate < target_acceptance_rate:
                    epsilon *= (1 - adapt_factor)
                else:
                    epsilon *= (1 + adapt_factor)
                adapt_count = 0
        else:
            print('hmc reject')

    if cov:
        Xs = R.T @ Xs + mu.reshape(mu.shape[0], 1)
    else:
        Xs = np.linalg.solve(R, Xs) + mu.reshape(mu.shape[0], 1)

    return Xs

