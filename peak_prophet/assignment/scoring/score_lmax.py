"""
score_lmax.py

Author: natelgrw
Last Edited: 11/10/2025

Lambda max scoring utilities. Uses exponential similarity on 
lambda max difference (nm) with a chosen sigma.
"""

from __future__ import annotations
import math


def exponential_lmax_score(lmax_pred: float, lmax_obs: float, sigma: float = 56.0, max_score: float = 1.0) -> float:
    """
    Exponential similarity on lambda max difference (nm).
    
    score = max_score * exp(-|lambda_m - lambda_p| / sigma_uv)
    
    Where sigma_uv is roughly the model's RMSE.
    
    Parameters
    ----------
    lmax_pred : float
        Predicted lambda max (nm)
    lmax_obs : float
        Observed lambda max (nm)
    sigma : float, default=5.0
        Scale parameter (typically model RMSE in nm)
    max_score : float, default=1.0
        Maximum score value
    
    Returns
    -------
    float
        Score in [0, max_score]
    """
    delta = abs(lmax_pred - lmax_obs)
    if sigma <= 0:
        return 0.0
    return float(max_score * math.exp(-delta / sigma))
