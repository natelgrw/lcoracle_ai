"""
score_ms.py

Author: natelgrw
Last Edited: 12/04/2025

Mass spectrum similarity scoring utilities. Uses fraction of matched 
predicted adduct masses to the observed m/z values.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np


def adduct_match_score(
    pred_adduct_masses: List[float] | np.ndarray | Dict[str, Tuple[float, float]],
    obs_mz: List[float] | np.ndarray,
    ppm: float = 5.0,
    mz_tol: Optional[float] = None,
) -> float:
    """
    Score based on weighted fraction of matched predicted adduct masses.
    
    f_ms = (sum of probabilities of matched predicted masses) / (sum of all predicted probabilities)
    
    If probabilities are not provided (list input), uses uniform weighting (1.0 per mass).
    
    Parameters
    ----------
    pred_adduct_masses : List[float], np.ndarray, or Dict[str, Tuple[float, float]]
        Predicted adduct m/z values or dict {adduct_name: (mass, prob)}
    obs_mz : List[float] or np.ndarray
        Observed m/z values from the measured spectrum
    ppm : float, default=5.0
        Parts-per-million tolerance for matching
    mz_tol : Optional[float]
        Absolute m/z tolerance in Da. If provided, used instead of ppm.
    
    Returns
    -------
    float in [0, 1]
        Weighted fraction of predicted masses that have a match in observed spectrum
    """
    # extract masses and probabilities from dict if needed
    if isinstance(pred_adduct_masses, dict):
        pred_masses = [mass for mass, _ in pred_adduct_masses.values()]
        pred_probs = [prob for _, prob in pred_adduct_masses.values()]
    else:
        pred_masses = list(pred_adduct_masses)
        pred_probs = [1.0] * len(pred_masses)
    
    pred_masses = np.asarray(pred_masses, dtype=float)
    pred_probs = np.asarray(pred_probs, dtype=float)
    obs_mz = np.asarray(obs_mz, dtype=float)
    
    if pred_masses.size == 0:
        return 0.0
    if obs_mz.size == 0:
        return 0.0
    
    # normalize probabilities so they sum to 1.0
    total_weight = pred_probs.sum()
    if total_weight <= 0:
        return 0.0
    pred_probs_normalized = pred_probs / total_weight
    
    obs_sorted = np.sort(obs_mz)
    matched_weight = 0.0
    used = np.zeros(obs_sorted.size, dtype=bool)
    
    for i, pred_mass in enumerate(pred_masses):
        tol = mz_tol if mz_tol is not None else (ppm * pred_mass / 1e6)
        idx = np.searchsorted(obs_sorted, pred_mass)
        candidates = []
        if idx < obs_sorted.size:
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        
        matched = False
        for cand_idx in candidates:
            if used[cand_idx]:
                continue
            delta = abs(obs_sorted[cand_idx] - pred_mass)
            if delta <= tol:
                matched = True
                used[cand_idx] = True
                break
        
        if matched:
            matched_weight += pred_probs_normalized[i]
    
    return float(matched_weight)
