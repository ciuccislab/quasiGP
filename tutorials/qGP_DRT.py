
#__authors__ = 'Francesco Ciucci, Adeleke Maradesa'

__date__ = '02 September, 2024'


## load necessary packages
import numpy as np
from math import sin, cos, pi
from math import pi, log, exp,sqrt
from scipy.optimize import minimize
from scipy.linalg import solve   
from numpy.linalg import cholesky
import importlib
import time
import HMC
importlib.reload(HMC)
##


def kernel(log_tau, psi, log_tau_prime, psi_prime, sigma_k, l_f, l_psi):
    """
    Calculate the kernel K(log_tau, psi, log_tau_prime, psi_prime) for given parameters.
    
    Inputs:
    - log_tau: log of tau
    - psi: parameter vector psi
    - log_tau_prime: log of tau prime
    - psi_prime: parameter vector psi prime
    - sigma_k: kernel scaling factor
    - l_f: length scale for log_tau
    - l_psi: length scale for psi

    Output:
    K_logtau_psi
    """
    # see equation 12 in the main article
    k_logtau = np.exp(-(log_tau - log_tau_prime)**2 / (2 * l_f**2))
    k_psi = np.exp(-(np.linalg.norm(psi - psi_prime)**2) / (2 * l_psi**2))
    k_logtau_psi = (sigma_k**2)*(k_logtau*k_psi)

    return k_logtau_psi

def compute_K(log_tau_vec, psi, psi_prime, sigma_k, l_f, l_psi):
    """
    Compute the kernel matrix K for a given set of log_tau values and parameters psi, psi_prime.

    Inputs:
    - log_tau_vec: vector of log_tau values
    - psi: parameter vector psi
    - psi_prime: parameter vector psi prime
    - sigma_k: kernel scaling factor
    - l_f: length scale for log_tau
    - l_psi: length scale for psi

    Output:
    K: computed kernel matrix K
    """
    N_taus = log_tau_vec.size
    out_K = np.zeros((N_taus, N_taus))

    for m in range(N_taus):
        for n in range(N_taus):
            k_logtau_psi = kernel(log_tau_vec[m], psi, log_tau_vec[n], \
                                     psi_prime, sigma_k, l_f, l_psi)
            out_K[m, n] = k_logtau_psi
    
    return out_K


def compute_A_re(freq_vec, tau_vec):
    """
    Compute the discretization matrix, A_re, for the real part of the impedance.
     
    Inputs:
    - freq_vec: vector of frequencies
    - tau_vec: vector of timescales
    
    Output:
    A_re: real-impedance-part-discretization matrix 
    """
    
    omega_vec = 2.*pi*freq_vec
    log_tau_vec = np.log(tau_vec)

    # number of elements in tau and freqs
    N_tau = tau_vec.size
    N_f = freq_vec.size

    # define output function
    out_A_re = np.zeros((N_f, N_tau))

    # integrand
    f_re = lambda omega, log_tau: 1./(1+(omega*exp(log_tau))**2)

    for m in range(0, N_f):
        
        for n in range(0, N_tau):
            
            if n == 0:
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 1/4*np.array([Delta_np1, Delta_np1])
                I_vec = np.array([f_re(omega_vec[m], log_tau_center), 
                                    f_re(omega_vec[m], log_tau_right)])

            elif n == N_tau-1:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1])
                I_vec = np.array([f_re(omega_vec[m], log_tau_left), 
                                    f_re(omega_vec[m], log_tau_center)])

            else:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1+Delta_np1, Delta_np1])
                I_vec = np.array([f_re(omega_vec[m], log_tau_left), 
                                    f_re(omega_vec[m], log_tau_center), 
                                    f_re(omega_vec[m], log_tau_right)])

            out_A_re[m,n] = np.dot(a_vec, I_vec)
            
    return out_A_re

def compute_A_im(freq_vec, tau_vec):
    """
    Compute the discretization matrix, A_im, for the imaginary part of the impedance.
     
    Inputs:
    - freq_vec: vector of frequencies
    - tau_vec: vector of timescales
    
    Output:
    A_im: imaginary-impedance-part-discretization matrix 
    """
    omega_vec = 2.*pi*freq_vec
    log_tau_vec = np.log(tau_vec)

    # number of elements in tau and freqs
    N_tau = tau_vec.size
    N_f = freq_vec.size

    # define output function
    out_A_im = np.zeros((N_f, N_tau))

    # integrand
    f_im = lambda omega, log_tau: -omega*exp(log_tau)/(1+(omega*exp(log_tau))**2)

    for m in range(0, N_f):
        
        for n in range(0, N_tau):
            
            if n == 0:
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 1/4*np.array([Delta_np1, Delta_np1])
                I_vec = np.array([f_im(omega_vec[m], log_tau_center), 
                                    f_im(omega_vec[m], log_tau_right)])

            elif n == N_tau-1:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1])
                I_vec = np.array([f_im(omega_vec[m], log_tau_left), 
                                    f_im(omega_vec[m], log_tau_center)])

            else:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1+Delta_np1, Delta_np1])
                I_vec = np.array([f_im(omega_vec[m], log_tau_left), 
                                    f_im(omega_vec[m], log_tau_center), 
                                    f_im(omega_vec[m], log_tau_right)])

            out_A_im[m,n] = np.dot(a_vec, I_vec)
            
    return out_A_im


def compute_Gamma(theta, log_tau_vec, psi_vec, N_psi):
    """
    Compute the Gamma matrix for multiple experiments (see equation 11 in the main article)

    Inputs:
    - theta: hyperparameter vector (sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R)
    - log_tau_vec: vector of log timescales
    - psi_vec: vector of experimental states
    - N_psi: Number of experimental conditions

    Output:
    - Gamma_exp: Computed Gamma matrix for all experiments
    """
    sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R = theta
    N_taus = log_tau_vec.size 

    Gamma = np.zeros(((N_taus+1) * N_psi, (N_taus+1) * N_psi))

    for k in range(N_psi):
        for l in range(k, N_psi):
            Gamma_loc = np.zeros((N_taus+1, N_taus+1))
            
            K = compute_K(log_tau_vec, psi_vec[k], psi_vec[l], sigma_k, l_f, l_psi)
            k_psi = kernel(1, psi_vec[k], 1, psi_vec[l], sigma_R, 1, l_R)
            
            Gamma_loc[0, 0] = k_psi
            Gamma_loc[1:, 1:] = K

            row_start, col_start = (N_taus+1) * k, (N_taus+1) * l
            Gamma[row_start:row_start + (N_taus+1), col_start:col_start + (N_taus+1)] = Gamma_loc
            Gamma[col_start:col_start + (N_taus+1), row_start:row_start + (N_taus+1)] = Gamma_loc.T
    
    return Gamma


def create_Z_exp_re_im(Z_exp_re, Z_exp_im, N_freqs):
    """
    Create the stacked real and imaginary impedance vectors for multiple experiments.

    Inputs:
    - Z_exp_re: list of real parts of experimental impedances
    - Z_exp_im: list of imaginary parts of experimental impedances
    - N_freqs: number of frequencies

    Output:
    Z_exp_re_im: stacked real and imaginary impedance vectors for all experiments
    """
    # number of experimental conditions
    N_psi = len(Z_exp_re)

    # initialize the stacked vector for real and imaginary parts
    Z_exp_re_im = np.zeros(2 * N_freqs * N_psi)

    for n in range(N_psi):
        # stacked real and imaginary vectors of the impedance
        start_idx = n * 2 * N_freqs
        Z_exp_re_im[start_idx:start_idx + N_freqs] = Z_exp_re[n]
        Z_exp_re_im[start_idx + N_freqs:start_idx + 2 * N_freqs] = Z_exp_im[n]

    return Z_exp_re_im


def compute_A_exp(freq_vecs, tau_vecs, N_psi):
    """
    Compute the discretization matrix A_exp for multiple experiments.

    Inputs:
    freq_vecs: frequency vectors for each experiment
    tau_vecs: timescale vectors for each experiment
    N_exp: number of experiments

    Output:
    A_exp: discretization matrix, A_exp, for all experiments
    """
    # number of timescales
    N_taus = tau_vecs[0].shape[0]
    # number of frequencies
    N_freqs = freq_vecs[0].shape[0]
    
    # initialize the combined A_exp matrix for all experiments
    A_exp = np.zeros((2 * N_taus * N_psi, (N_taus + 1) * N_psi))
    
    for k in range(N_psi):
        # compute A for the current experiment
        A = np.zeros((2 * N_freqs, N_taus + 1))
        
        # compute the real, A_re, and imaginary, A_im,  parts of A
        A_re = compute_A_re(freq_vecs[k], tau_vecs[k])
        A_im = compute_A_im(freq_vecs[k], tau_vecs[k])
        
        # real part
        A[:N_freqs, 0] = 1.0 
        A[:N_freqs, 1:] = A_re
        
        # imaginary part
        A[N_freqs:, 1:] = A_im
        
        # calculate the row and column indexing for placing A into A_exp
        row_start, row_end = 2 * N_taus * k, 2 * N_taus * (k + 1)
        col_start, col_end = (N_taus + 1) * k, (N_taus + 1) * (k + 1)
        
        # place A in the correct position in A_exp
        A_exp[row_start:row_end, col_start:col_end] = A
    
    return A_exp


def Sherman_Morrison_Woodbury(Gamma_exp, A_exp, N_freqs, N_psi, sigma_n, r = 160):
    """
    Apply the Sherman-Morrison-Woodbury formula to compute the inverse of (A + UCV^T).

    Inputs:
        - Gamma_exp : the Gamma matrix for N_exp.
        - A_exp : the discretization matrix for N_exp.
        - N_freqs : the number of frequencies.
        - N_exp : the number of experiments
        - sigma_n : the noise standard deviation
        - r : the number of singular values to retain.

    Outputs:
        The inverse of (A + UCV^T) as per the Sherman-Morrison-Woodbury formula.
    """
    
    # compute Gamma
    Gamma = A_exp @ (Gamma_exp @ A_exp.T)
    
    # construct matrix A
    A = (sigma_n**2) * np.eye(2 * N_freqs * N_psi)
    
    # singular value decomposition (SVD)
    U, sigma, Vt = np.linalg.svd(Gamma + 1E-9 * np.eye(*Gamma.shape), full_matrices=False)
    
    # retain only the top r singular values and vectors
    U_r, sigma_r, Vt_r = U[:, :r], np.diag(sigma[:r]), Vt[:r, :]
    
    # compute A_inv
    A_inv = (sigma_n**-2) * np.eye(2 * N_freqs * N_psi)
    
    # calculate the diagonal elements for Sigma
    sigma_diag = sigma[:r]
    
    # compute the diagonal matrix as per the equation (15)
    diag_matrix = np.diag((sigma[:r] * sigma_n**2) / (sigma[:r] + sigma_n**2))
    
    # apply the Sherman-Morrison-Woodbury formula (see equation 14 in the main article)
    Gamma_inv = (sigma_n**-2) * np.eye(2 * N_freqs * N_psi) - (sigma_n**-4) * U_r @ diag_matrix @ Vt_r
    
    # compute the detrminant (see equation 13 in the main article)
    Gamma_det = np.sum(np.log(sigma_n**2 + sigma))
    
    return Gamma_inv, Gamma_det



def NMLL_fct_exp(theta, A_exp, Z_exp_re_im, N_freqs, log_tau_vec, psi_vec, N_psi): 
    """
    Compute the negative marginal log-likelihood.
    
    Inputs:
    - theta: hyperparameter vector (sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R)
    - A_exp: discretization matrix for N_exp
    - z_exp_re_im: stacked vectors of experimental impedance for N_exp
    - N_freqs: number of frequencies
    - log_tau_vec: vector of log timescales
    - psi_vec: vector of experimental states
    - N_exp: number of experiments

    Output:
    nmll: negative marginal log-likelihood
    """
    # extract hyperparameters
    sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R = theta

    # compute Gamma_exp matrix
    Gamma_exp = compute_Gamma(theta, log_tau_vec, psi_vec, N_psi)
    Psi = A_exp @ (Gamma_exp @ A_exp.T) + (sigma_n**2) * np.eye(2 * N_freqs * N_psi)
    
    # use Sherman-Morrison-Woodbury formula to compute the inverse of Psi
    Psi_inv,Gamma_det = Sherman_Morrison_Woodbury(Gamma_exp, A_exp, N_freqs, N_psi, sigma_n)
    
    # compute alpha
    alpha = Psi_inv @ Z_exp_re_im
    
    # compute the negative marginal log-likelihood (see equation 13 in the main article)
    nmll = 0.5 * np.dot(Z_exp_re_im, alpha) + 0.5 * Gamma_det + 0.5 * len(Z_exp_re_im) * np.log(2 * np.pi)
    
    return nmll



def compute_mux_and_sigmax(theta, A_exp, N_freqs, Z_exp_re_im, log_tau_vec, psi_vec, N_psi):
    """
    Compute the posterior mean, mu_x_given_Z, and posterior covariance matrix for N_exp experiments.

    Inputs:
    - theta: hyperparameter vector (sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R)
    - A_exp: discretization matrix
    - N_freqs: number of frequencies
    - Z_exp_re_im: vector of stacked real and imaginary parts of the experimental impedances
    - log_tau_vec: vector of log timescales
    - psi_vec: vector of experimental states
    - N_exp: number of experiments

    Outputs:
    - mu_x_given_Z: posterior mean vector
    - Sigma_x_given_Z: posterior covariance matrix
    """
    
    # optimized parameters
    sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R = theta

    # compute Gamma_exp matrix
    Gamma_exp = (compute_Gamma(theta, log_tau_vec, psi_vec, N_psi))
    
    # use Sherman-Morrison-Woodbury formula to compute the inverse of Psi
    Psi_inv,_ = Sherman_Morrison_Woodbury(Gamma_exp, A_exp, N_freqs, N_psi, sigma_n)
    # compute Xi matrix
    Xi = Gamma_exp @ A_exp.T
    
    # solve for alpha
    alpha = Psi_inv @ Z_exp_re_im

    # compute posterior mean, mu_x_given_Z (see equation 10a in the main article)
    mu_x_given_Z = Xi @ alpha

    # compute posterior covariance matrix, Sigma_x_given_Z (see equation 10a in the main article)
    Sigma_x_given_Z = Gamma_exp - Xi @ (Psi_inv @ Xi.T)

    return mu_x_given_Z, Sigma_x_given_Z


def sample_DRT(theta, A_exp, Z_exp_re_im, N_freqs, log_tau_vec, psi_vec, N_psi):
    """
    Perform hyperparameter optimization and sampling DRT from truncated multinormal distribution using Hamiltonian Monte Carlo sampler.

    Inputs:
    - theta: hyperparameter vector (sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R)
    - A_exp: discretization matrix for N_exp
    - Z_exp_re_im: vector of stacked real and imaginary parts of the experimental impedances
    - N_freqs: number of frequencies
    - log_tau_vec: vector of log timescales
    - psi_vec: vector of experimental states
    - N_exp: number of experiments

    Outputs:
    - elapsed_time: time taken for training quasi-GP framework
    - samples_raw: vector of DRTs sampled via Hamiltonian Monte Carlo sampler
    """
    # number of log timesacles
    N_taus = log_tau_vec.size
    
    # start time for training
    start_time = time.time()
    
    def print_results(theta):
        print(' '.join(f'{param:.7f}' for param in theta))
    
    # define the bounds for hyperparameters when using SLSQP, and remove bound for other optimizer such as Powell
    bounds = ((0.3, 0.6), (3.0, 10.0), (0.5, 10.0), (0.7, 1.0), (0.8, 1.0), (0, 1.0))
    
    print('sigma_n, sigma_R, sigma_k, l_f, l_psi, l_R')
    
    # optimize hyperparameters (Powell or SLSQP can be used as optimizer) 
    # Note: always remember to remove bound constraint for Powell.
    res = minimize(NMLL_fct_exp, theta, args=(A_exp, Z_exp_re_im, N_freqs, log_tau_vec, psi_vec, N_psi),
                   method='SLSQP', callback=print_results, options={'disp': True}, bounds = bounds) # 
    
    theta_opt = res.x

    # compute the posterior mean and covariance
    mu_x_given_Z, Sigma_x_given_Z = compute_mux_and_sigmax(theta_opt, A_exp, N_freqs,Z_exp_re_im,log_tau_vec, psi_vec, N_psi)
    
    # inputs HMC sampler
    # F*X+g >0 for non-negativity constraint
    F = np.eye((N_taus + 1) * N_psi)
    g = np.zeros((N_taus + 1) * N_psi)
    # covariance & mean
    M = Sigma_x_given_Z + 1E-8 * np.eye((N_taus + 1) * N_psi)
    mu_r = mu_x_given_Z
    
    print("\n==========================================================\n")
    print("Sampling from truncated multivariate normal using Hamiltonian Monte Carlo.....")
    print("\n==========================================================\n")
    
    # initial val
    initial_x_given_Z = np.abs(mu_x_given_Z)
    mu_r = mu_x_given_Z
    #
    samples_raw = HMC.generate_tmg(F, g, M, mu_r, initial_x_given_Z, cov=True, L=10000)
    ## end time
    elapsed_time = time.time() - start_time
    
    return elapsed_time, samples_raw


def gamma_results(N_taus, samples_gamma, N_psi):
    
    """
    Extract the recovered DRTs and their corresponding upper and lower bands.

    Inputs:
    - N_taus: number of timescales
    - samples_gamma: vector of sampled DRT values
    - N_exp: number of experiments

    Outputs:
    - gamma_median: vector of recovered DRT for each experiment
    - gamma_upper: vector of upper credible bands of the recovered DRT for each experiment
    - gamma_lower: vector of lower credible bands of the recovered DRT for each experiment
    """
    
    # remove burn-in samples
    samples = samples_gamma[:, 1000:]
    # slice out only samples of gamma
    gamma_samples = samples[1:, :]
    
    # compute median of the recovered DRTs and percentiles for all N_exp
    gamma = np.nanmedian(gamma_samples, axis=1)
    gamma_percentile1 = np.percentile(gamma_samples, 1, axis=1)
    gamma_percentile9 = np.percentile(gamma_samples, 99, axis=1)
    
    # initialize lists to store results
    gamma_median, gamma_upper_band, gamma_lower_band = [], [], []
        
    for k in range(N_psi):
        
        start_index = k * (N_taus + 1)
        end_index = start_index + N_taus
        
        gamma_median.append(gamma[start_index:end_index])
        gamma_upper_band.append(gamma_percentile9[start_index:end_index])
        gamma_lower_band.append(gamma_percentile1[start_index:end_index])
        
    return gamma_median, gamma_upper_band, gamma_lower_band


def recovered_impedances(N_freqs, samples_gamma, A_exp, N_psi):

    """
    Extract the recovered impedances and their credible bounds.

    Inputs:
    - N_freqs: number of frequcies
    - samples_gamma: vector of sampled DRT values
    - A_exp: discretization matrix for N_exp experiments
    - N_exp: number of experiments

    Outputs:
    - Z_re_median: vector of the recovered real part of the impedances for each experiment
    - Z_im_median: vector of the recovered imaginary part of the impedances for each experiment
    - Z_re_upper: vector of the recovered upper-credible bands for the real part of the impedances 
    - Z_im_upper: vector of the recovered upper-credible bands for the imaginary part of the impedances
    - Z_re_lower: vector of the recovered lower-credible bands for the real part of the impedances
    - Z_im_lower: vector of the recovered lower-credible bands for the imaginary part of the impedances
    """
    
    # remove burn-in samples and slice out only samples of gamma
    gamma_samples = samples_gamma[:, 1000:]
    samples_Z_re_im = A_exp @ gamma_samples

    # compute the median and percentiles of the samples
    Z_re_im_median = np.nanmedian(samples_Z_re_im, axis=1)
    Z_re_im_percentile1 = np.percentile(samples_Z_re_im, 1, axis=1)
    Z_re_im_percentile99 = np.percentile(samples_Z_re_im, 99, axis=1)
    
    # initialize lists to store results
    Z_re_median, Z_im_median = [], []
    Z_re_upper, Z_im_upper = [], []
    Z_re_lower, Z_im_lower = [], []
    
    for k in range(N_psi):
        re_start_idx = 2 * k * N_freqs
        re_end_idx = (2 * k + 1) * N_freqs
        im_start_idx = re_end_idx
        im_end_idx = (2 * k + 2) * N_freqs

        # extract the impedance values for the current experiment and store them appropriately in the designated lists
        Z_re_median.append(Z_re_im_median[re_start_idx:re_end_idx])
        Z_im_median.append(Z_re_im_median[im_start_idx:im_end_idx])
        Z_re_upper.append(Z_re_im_percentile99[re_start_idx:re_end_idx])
        Z_im_upper.append(Z_re_im_percentile99[im_start_idx:im_end_idx])
        Z_re_lower.append(Z_re_im_percentile1[re_start_idx:re_end_idx])
        Z_im_lower.append(Z_re_im_percentile1[im_start_idx:im_end_idx])

    return Z_re_median, Z_im_median, Z_re_upper, Z_im_upper, Z_re_lower, Z_im_lower
