"""
PPI++ point estimator and optimal lambda for AutoEval under dependent data.

Reference: Boyeau et al. (2025), AutoEval Done Right.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def ppi_estimate(phi_lab, syn_lab, syn_unl, lam=None):
    """
    Compute the PPI++ point estimate, variance, and optimal lambda.

    Parameters
    ----------
    phi_lab : np.ndarray, shape (n,)
        True label-based scores on the labeled split.
    syn_lab : np.ndarray, shape (n,)
        Synthetic annotator scores on the labeled split.
    syn_unl : np.ndarray, shape (N,)
        Synthetic annotator scores on the unlabeled split.
    lam : float or None
        If provided, use this lambda instead of solving for the optimum.

    Returns
    -------
    mu_hat : float
        PPI++ point estimate of the mean.
    var_hat : float
        Estimated variance of mu_hat (scalar).
    lam_opt : float
        Lambda used (optimal if lam=None, else the supplied value).
    """
    n = len(phi_lab)
    N = len(syn_unl)

    mu_syn_unl = np.mean(syn_unl)
    diff = phi_lab - syn_lab  # rectifier

    if lam is None:
        # Optimal lambda minimises asymptotic variance
        cov_diff_syn = np.cov(diff, syn_lab, ddof=1)
        var_syn = np.var(syn_lab, ddof=1)

        if var_syn < 1e-12:
            lam_opt = 1.0
        else:
            lam_opt = float(np.clip(
                1.0 - cov_diff_syn[0, 1] / var_syn, 0.0, 2.0
            ))
    else:
        lam_opt = float(lam)

    mu_hat = float(np.mean(phi_lab) + lam_opt * (mu_syn_unl - np.mean(syn_lab)))

    # Variance: Var(mean(phi_lab)) + lam^2 * Var(mean(syn_lab)) - 2*lam*Cov
    rectifier_term = phi_lab - lam_opt * syn_lab
    var_hat = float(np.var(rectifier_term, ddof=1) / n
                    + (lam_opt ** 2) * np.var(syn_unl, ddof=1) / N)

    return mu_hat, var_hat, lam_opt