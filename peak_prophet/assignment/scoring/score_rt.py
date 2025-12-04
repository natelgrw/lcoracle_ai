"""
score_rt.py

Author: natelgrw
Last Edited: 12/04/2025

Retention time scoring utilities using relative order scoring
and exponential scoring
"""

from __future__ import annotations
from typing import Optional, List
import math


def relative_order_rt_score(
    rt_pred: float,
    rt_obs: float,
    all_predicted_rts: List[float],
    all_observed_rts: List[float],
    method_length: Optional[float] = None,
    max_score: float = 1.0
) -> float:
    """
    Score based on relative order preservation (rank-based scoring).
    
    Computes how well the predicted RT preserves the relative order of observed RTs.
    Uses normalized rank positions within the method.
    
    Parameters
    ----------
    rt_pred : float
        Predicted retention time (in minutes)
    rt_obs : float
        Observed retention time (in minutes)
    all_predicted_rts : List[float]
        All predicted RTs (for computing relative position)
    all_observed_rts : List[float]
        All observed RTs (for computing relative position)
    method_length : Optional[float]
        Total method length in minutes. If None, uses max of all RTs.
    max_score : float, default=1.0
        Maximum score value
    
    Returns
    -------
    float
        Score in [0, max_score] based on rank preservation
    """
    if rt_pred is None or rt_obs is None:
        return 0.0
    
    # filter out None values
    valid_pred = [r for r in all_predicted_rts if r is not None]
    valid_obs = [r for r in all_observed_rts if r is not None]
    
    # compute min and max for normalization (handles negative RTs)
    min_pred = min(valid_pred)
    max_pred = max(valid_pred)
    min_obs = min(valid_obs)
    max_obs = max(valid_obs)
    
    # determine normalization range
    if method_length is None:
        min_rt = min(min_pred, min_obs)
        max_rt = max(max_pred, max_obs)
    else:
        min_rt = min(min_pred, min_obs)
        max_rt = method_length
    
    # normalize RTs to [0, 1] range
    range_rt = max_rt - min_rt
    if range_rt <= 0:
        return 0.0
    
    pred_norm = (rt_pred - min_rt) / range_rt
    obs_norm = (rt_obs - min_rt) / range_rt
    
    # compute relative ranks in percentile positions
    pred_rank = sum(1 for r in valid_pred if r < rt_pred) / len(valid_pred)
    obs_rank = sum(1 for r in valid_obs if r < rt_obs) / len(valid_obs)
    
    rank_diff = abs(pred_rank - obs_rank)
    norm_diff = abs(pred_norm - obs_norm)
    
    rank_score = max_score * (1.0 - rank_diff)
    position_score = max_score * (1.0 - norm_diff)
    
    return float(0.7 * rank_score + 0.3 * position_score)


def exponential_rt_score(rt_pred: float, rt_obs: float, sigma: float = 0.3, max_score: float = 1.0) -> float:
    """
    Exponential similarity on retention time difference (absolute scoring).
    
    score = max_score * exp(-|RT_m - RT_p| / sigma_rt)
    
    Parameters
    ----------
    rt_pred : float
        Predicted retention time (in minutes)
    rt_obs : float
        Observed retention time (in minutes)
    sigma : float, default=0.3
        Scale parameter (typically model MAE in minutes)
    max_score : float, default=1.0
        Maximum score value
    
    Returns
    -------
    float
        Score in [0, max_score]
    """
    delta = abs(rt_pred - rt_obs)
    if sigma <= 0:
        return 0.0
    return float(max_score * math.exp(-delta / sigma))
