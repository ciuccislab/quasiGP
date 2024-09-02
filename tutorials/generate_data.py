import numpy as np
from math import sin, cos, pi, sqrt


def generate_sythentic_impedance_data(param, sigma_n_exp_0, tau_vec, freq_vec, tau_vec_plot, N_exp, N_zarc):
    """
    Generate experimental and exact impedance data for a given ZARC model.
    
    Inputs:
    - param: parameter vector for the ZARC model
    - sigma_n_exp_0: initial experimental noise parameter
    - tau_vec: vector of timescales
    - freq_vec: vector of frequencies
    - tau_vec_plot: vector of timescale used for plotting
    - N_exp: number of experimental points
    
    Outputs:
    - Z_exp_re: vector of the real part of the experimental impedances
    - Z_exp_im: vector of the imaginary part of the experimental impedances
    - Z_exact_re: vector of the real part of the exact impedances
    - Z_exact_im: vector of the imaginary part of the exact impedance
    - gamma_exact: vector of the exact DRT values
    - gamma_exact_plot: vector of the exact DRT values used for plotting
    - psi_vec: vector of experimetal states
    """
    num_params = len(param)
    psi_vec = np.linspace(1.0, 0.1, num=N_exp, endpoint=True)

    Z_exp_re = [0] * N_exp
    Z_exp_im = [0] * N_exp
    Z_exact_re = [0] * N_exp
    Z_exact_im = [0] * N_exp
    gamma_exact = [0] * N_exp
    gamma_exact_plot = [0] * N_exp
    
    
    for n in range(N_exp):
        R_inf = param[0]
        Z_exact_loc = R_inf
        gamma_exact_loc = np.zeros_like(tau_vec)
        gamma_exact_plot_loc = np.zeros_like(tau_vec_plot)

        for m in range(N_zarc):
            idx = 1 + m * 3
            R_ct = param[idx] * psi_vec[n]
            phi = param[idx + 1]
            tau = param[idx + 2]

            T_0 = tau**phi / R_ct
            Z_exact = 1./(1./R_ct + T_0*(1j*2.*np.pi*freq_vec)**phi)

            gamma_loc = (R_ct)/(2.*np.pi)*np.sin((1.-phi)*np.pi)/(np.cosh(phi*np.log(tau_vec/tau))-np.cos((1.-phi)*np.pi))
            gamma_loc_plot = (R_ct)/(2.*np.pi)*np.sin((1.-phi)*np.pi)/(np.cosh(phi*np.log(tau_vec_plot/tau))-\
                                                                       np.cos((1.-phi)*np.pi))

            Z_exact_loc += Z_exact
            gamma_exact_loc += gamma_loc
            gamma_exact_plot_loc += gamma_loc_plot
            
        #seed for reproducibility
        np.random.seed(12129)
        sigma_n_exp = sigma_n_exp_0 * np.sqrt(psi_vec[n])
        Z_exp = Z_exact_loc + (sigma_n_exp ** 2) * (np.random.normal(0, 1, Z_exact_loc.shape) + 1j * \
                                                      np.random.normal(0, 1, Z_exact_loc.shape))

        Z_exp_re[n], Z_exp_im[n] = np.real(Z_exp), np.imag(Z_exp)
        Z_exact_re[n], Z_exact_im[n] = np.real(Z_exact_loc), np.imag(Z_exact_loc)
        gamma_exact[n] = gamma_exact_loc
        gamma_exact_plot[n] = gamma_exact_plot_loc

    return Z_exp_re, Z_exp_im, Z_exact_re, Z_exact_im, gamma_exact, gamma_exact_plot, psi_vec




def generate_freq_and_tau_vec(N_exp, N_freqs, N_taus, log_freq_min, log_freq_max, log_tau_min, log_tau_max):
    """
    Generate frequency and tau vectors for experiments.
    
    Inputs:
    - N_exp: number of experiments.
    - N_freqs: number of frequencies
    - N_taus: number of timescales.
    - freq_min: minimum value of frequencies.
    - freq_max: maximum value of frequencies.
    - log_tau_min: minimum value of log timescales.
    - log_tau_max: maximum value of log timescales.
    
    Outputs:
    - freq_vecs: frequency vectors for all experiments.
    - tau_vecs:  timescale vectors for all experiments.
    - log_tau_vecs: log timescale vectors for all experiments.
    """
    freq_vecs = []
    tau_vecs = []
    log_tau_vecs = []

    for i in range(N_exp):
        # Frequency vector for each experiment
        freq_vec = np.logspace(log_freq_min, log_freq_max, num=N_freqs, endpoint=True)
        # timescale vector for each experiment
        tau_vec = np.logspace(log_tau_min, log_tau_max, num=N_taus, endpoint=True)
        # Log timescale vector for each experiment
        log_tau_vec = np.log(tau_vec)

        freq_vecs.append(freq_vec)
        tau_vecs.append(tau_vec)
        log_tau_vecs.append(log_tau_vec)

    return freq_vecs, tau_vecs, log_tau_vecs
