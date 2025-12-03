"""
score_aggregate.py

Author: natelgrw
Last Edited: 11/15/2025

Aggregate scoring utilities. Uses relative order scoring for RT, 
exponential scoring for UV and MS, and prior probability from ASKCOS.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Union
import numpy as np

from .score_ms import adduct_match_score
from .score_rt import relative_order_rt_score, exponential_rt_score
from .score_lmax import exponential_lmax_score

# importing PeakProphet custom classes
from decoding.LCMS_meas_man import LCMSMeasMan
from decoding.LCMSUV_meas_man import LCMSUVMeasMan
from predictions.utils.rxn_classes import ChemicalReaction

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ===== Cumulative Scoring Function ===== #


def score_peak_candidate(
    peak_rt: float,
    peak_mz: List[float],
    peak_lmax: Optional[float],
    candidate_rt: Optional[float],
    candidate_ms_values: Optional[Dict[str, Tuple[float, float]]],
    candidate_lmax: Optional[float],
    candidate_prior: Optional[float],
    all_predicted_rts: Optional[List[float]] = None,
    all_observed_rts: Optional[List[float]] = None,
    method_length: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    rt_sigma: float = 0.3,
    lmax_sigma: float = 5.0,
    ms_ppm: float = 5.0,
    use_relative_rt: bool = True,
) -> float:
    """
    Compute combined score for a single peak-candidate pair.
    
    Score = w_rt * f_rt + w_uv * f_uv + w_ms * f_ms + w_prior * f_prior
    
    Parameters
    ----------
    peak_rt : float
        Observed retention time (minutes)
    peak_mz : List[float]
        Observed m/z values
    peak_lmax : Optional[float]
        Observed lambda max (nm) or None
    candidate_rt : Optional[float]
        Predicted retention time (minutes) or None
    candidate_ms_values : Optional[Dict[str, Tuple[float, float]]]
        Predicted adduct masses {adduct_name: (mass, prob)} or None
    candidate_lmax : Optional[float]
        Predicted lambda max (nm) or None
    candidate_prior : Optional[float]
        Prior probability from ASKCOS (0-1) or None
    all_predicted_rts : Optional[List[float]]
        All predicted RTs for relative order scoring
    all_observed_rts : Optional[List[float]]
        All observed RTs for relative order scoring
    method_length : Optional[float]
        Total method length (minutes) for relative RT scoring
    weights : Optional[Dict[str, float]]
        Weights for each component.
    rt_sigma : float, default=0.3
        sigma_rt for absolute RT scoring (minutes)
    lmax_sigma : float, default=5.0
        sigma_uv for UV scoring (nm)
    ms_ppm : float, default=5.0
        Parts-per-million tolerance for MS matching
    use_relative_rt : bool, default=True
        If True, use relative order RT scoring (requires all RTs)
    
    Returns
    -------
    float
        Combined score (can exceed 1.0 if weights sum > 1.0)
    """
    if weights is None:
        weights = {"rt": 0.2, "uv": 0.2, "ms": 0.4, "prior": 0.2}
    
    score = 0.0
    
    # retention time score
    if candidate_rt is not None:
        if use_relative_rt and all_predicted_rts is not None and all_observed_rts is not None:
            rt_score = relative_order_rt_score(
                rt_pred=candidate_rt,
                rt_obs=peak_rt,
                all_predicted_rts=all_predicted_rts,
                all_observed_rts=all_observed_rts,
                method_length=method_length
            )
        else:
            rt_score = exponential_rt_score(candidate_rt, peak_rt, sigma=rt_sigma)
        score += weights.get("rt", 0.0) * rt_score
    
    # lambda max score
    if peak_lmax is not None and candidate_lmax is not None:
        uv_score = exponential_lmax_score(candidate_lmax, peak_lmax, sigma=lmax_sigma)
        score += weights.get("uv", 0.0) * uv_score
    
    # mass spec score
    if peak_mz and candidate_ms_values:
        ms_score = adduct_match_score(candidate_ms_values, peak_mz, ppm=ms_ppm)
        score += weights.get("ms", 0.0) * ms_score
    
    # prior probability score
    if candidate_prior is not None:
        score += weights.get("prior", 0.0) * float(candidate_prior)
    
    return float(score)


# ===== Candidate Ranking Function ===== #


def rank_candidates_for_peak(
    reaction: ChemicalReaction,
    meas_man: Union[LCMSMeasMan, LCMSUVMeasMan],
    peak_rt: float,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    rt_sigma: float = 0.3,
    lmax_sigma: float = 5.0,
    ms_ppm: float = 5.0,
    use_relative_rt: bool = True,
) -> List[Tuple[Dict, float]]:
    """
    Rank candidate compounds for a detected peak and return top-K recommendations.
    
    Parameters
    ----------
    reaction : ChemicalReaction
        ChemicalReaction object with predicted products
    meas_man : LCMSMeasMan or LCMSUVMeasMan
        Measurement manager with detected peaks
    peak_rt : float
        Apex retention time of the peak to score (minutes)
    top_k : int, default=5
        Number of top candidates to return
    weights : Optional[Dict[str, float]]
        Weights for scoring components
    rt_sigma : float, default=0.3
        sigma_rt for absolute RT scoring (minutes)
    lmax_sigma : float, default=5.0
        sigma_uv for UV scoring (nm)
    ms_ppm : float, default=5.0
        Parts-per-million tolerance for MS matching
    use_relative_rt : bool, default=True
        If True, use relative order RT scoring
    
    Returns
    -------
    List[Tuple[Dict, float]]
        List of (candidate_info, score) tuples, sorted by score descending
    """
    if peak_rt not in meas_man.peaks:
        return []
    
    peak_data = meas_man.peaks[peak_rt]
    peak_mz = [mz for mz, _ in peak_data.get('ms_spectrum', [])]
    peak_lmax = peak_data.get('lambda_max')
    
    # get all products
    products = reaction.get_products()
    if not products:
        return []
        
    # extract all RTs for relative order scoring
    # Products have RT in seconds, convert to minutes
    all_predicted_rts = [p.get_retention_time() / 60.0 if p.get_retention_time() else None 
                         for p in products]
    # Peak RTs are in seconds (from mocca2), convert to minutes for scoring
    all_observed_rts = [rt / 60.0 for rt in meas_man.peaks.keys()] if use_relative_rt else None
    
    # get method length
    method_length = None
    if reaction.lcms_gradient:
        method_length = max(t for t, _ in reaction.lcms_gradient)
    
    # Convert peak_rt from seconds to minutes for scoring
    peak_rt_minutes = peak_rt / 60.0
    
    # score each candidate
    scored = []
    for product in products:
        candidate_rt = product.get_retention_time() / 60.0 if product.get_retention_time() else None  # Convert to minutes
        candidate_ms = product.get_ms_values()
        candidate_lmax = product.get_lambda_max()
        candidate_prior = product.get_probability()
        
        score = score_peak_candidate(
            peak_rt=peak_rt_minutes,
            peak_mz=peak_mz,
            peak_lmax=peak_lmax,
            candidate_rt=candidate_rt,
            candidate_ms_values=candidate_ms,
            candidate_lmax=candidate_lmax,
            candidate_prior=candidate_prior,
            all_predicted_rts=all_predicted_rts,
            all_observed_rts=all_observed_rts,
            method_length=method_length,
            weights=weights,
            rt_sigma=rt_sigma,
            lmax_sigma=lmax_sigma,
            ms_ppm=ms_ppm,
            use_relative_rt=use_relative_rt
        )
        
        candidate_info = {
            'smiles': product.get_smiles(),
            'rt': candidate_rt,
            'lmax': candidate_lmax,
            'ms_values': candidate_ms,
            'prior': candidate_prior
        }
        scored.append((candidate_info, score))
    
    # sort by score in descending order
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ===== All Peaks Scoring Function ===== #


def score_all_peaks(
    reaction: ChemicalReaction,
    meas_man: Union[LCMSMeasMan, LCMSUVMeasMan],
    weights: Optional[Dict[str, float]] = None,
    rt_sigma: float = 0.3,
    lmax_sigma: float = 5.0,
    ms_ppm: float = 5.0,
    use_relative_rt: bool = True,
) -> np.ndarray:
    """
    Build score matrix for all peaks vs all candidates.
    
    Parameters
    ----------
    reaction : ChemicalReaction
        ChemicalReaction object with predicted products
    meas_man : LCMSMeasMan or LCMSUVMeasMan
        Measurement manager with detected peaks
    weights : Optional[Dict[str, float]]
        Weights for scoring components
    rt_sigma : float, default=0.3
        sigma_rt for absolute RT scoring (minutes)
    lmax_sigma : float, default=5.0
        sigma_uv for UV scoring (nm)
    ms_ppm : float, default=5.0
        Parts-per-million tolerance for MS matching
    use_relative_rt : bool, default=True
        If True, use relative order RT scoring
    
    Returns
    -------
    np.ndarray
        Score matrix of shape [n_candidates, n_peaks]
    """
    products = reaction.get_products()
    peak_rts = sorted(meas_man.peaks.keys())
    
    if not products or not peak_rts:
        return np.zeros((len(products), len(peak_rts)))
    
    # Extract all RTs for relative order scoring
    # Products have RT in seconds, convert to minutes
    all_predicted_rts = [p.get_retention_time() / 60.0 if p.get_retention_time() else None 
                         for p in products]
    # Peak RTs are in seconds (from mocca2), convert to minutes for scoring
    all_observed_rts = [rt / 60.0 for rt in peak_rts] if use_relative_rt else None
    
    # Get method length
    method_length = None
    if reaction.lcms_gradient:
        method_length = max(t for t, _ in reaction.lcms_gradient)
    
    # Build score matrix
    score_matrix = np.zeros((len(products), len(peak_rts)))
    
    for i, product in enumerate(products):
        candidate_rt = product.get_retention_time() / 60.0 if product.get_retention_time() else None
        candidate_ms = product.get_ms_values()
        candidate_lmax = product.get_lambda_max()
        candidate_prior = product.get_probability()
        
        for j, peak_rt in enumerate(peak_rts):
            peak_data = meas_man.peaks[peak_rt]
            peak_mz = [mz for mz, _ in peak_data.get('ms_spectrum', [])]
            peak_lmax = peak_data.get('lambda_max')  # None for LCMSMeasMan, float for LCMSUVMeasMan
            
            # Convert peak_rt from seconds to minutes for scoring
            peak_rt_minutes = peak_rt / 60.0
            
            score = score_peak_candidate(
                peak_rt=peak_rt_minutes,
                peak_mz=peak_mz,
                peak_lmax=peak_lmax,
                candidate_rt=candidate_rt,
                candidate_ms_values=candidate_ms,
                candidate_lmax=candidate_lmax,
                candidate_prior=candidate_prior,
                all_predicted_rts=all_predicted_rts,
                all_observed_rts=all_observed_rts,
                method_length=method_length,
                weights=weights,
                rt_sigma=rt_sigma,
                lmax_sigma=lmax_sigma,
                ms_ppm=ms_ppm,
                use_relative_rt=use_relative_rt
            )
            score_matrix[i, j] = score
    
    return score_matrix


def optimal_assignment(
    score_matrix: np.ndarray,
    min_score: float = 0.35
) -> Dict[int, Optional[int]]:
    """
    Compute optimal 1-1 assignment using Hungarian algorithm.
    Assigns None to peaks with no good matches (score < min_score).
    
    Parameters
    ----------
    score_matrix : np.ndarray
        Score matrix of shape [n_candidates, n_peaks]
    min_score : float, default=0.0
        Minimum score threshold for assignment. Peaks with best score < min_score get None.
    
    Returns
    -------
    Dict[int, Optional[int]]
        Dictionary mapping peak_index -> candidate_index (or None if no good match)
    """
    S = np.asarray(score_matrix, dtype=float)
    n_peaks = S.shape[1] if S.size > 0 else 0
    
    # Initialize all peaks to None
    assignment = {peak_idx: None for peak_idx in range(n_peaks)}
    
    if S.size == 0:
        return assignment
    
    if _HAS_SCIPY:
        cost = 1.0 - S
        r, c = linear_sum_assignment(cost)
        
        # Only assign if score meets threshold
        for cand_idx, peak_idx in zip(r, c):
            score = float(S[cand_idx, peak_idx])
            if score >= min_score:
                assignment[int(peak_idx)] = int(cand_idx)
        
        return assignment
    
    # greedy fallback
    S_copy = S.copy()
    while True:
        idx = np.unravel_index(np.argmax(S_copy), S_copy.shape)
        i, j = int(idx[0]), int(idx[1])
        best = float(S_copy[i, j])
        if best < min_score:
            break
        assignment[j] = i
        S_copy[i, :] = -np.inf
        S_copy[:, j] = -np.inf
        if not np.isfinite(S_copy).any():
            break
    
    return assignment


def optimal_assignment_with_top_k(
    score_matrix: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.4
) -> Tuple[Dict[int, Optional[int]], Dict[int, List[Tuple[int, float]]], float]:
    """
    Compute optimal 1-1 assignment AND top-K candidates for each peak.
    Assigns None to peaks with no good matches (score < min_score).
    
    Parameters
    ----------
    score_matrix : np.ndarray
        Score matrix of shape [n_candidates, n_peaks]
    top_k : int, default=5
        Number of top candidates to return for each peak
    min_score : float, default=0.0
        Minimum score threshold for assignment. Peaks with best score < min_score get None.
    
    Returns
    -------
    Tuple containing:
        assignment : Dict[int, Optional[int]]
            Dictionary mapping peak_index -> candidate_index (or None if no good match)
        top_k_per_peak : Dict[int, List[Tuple[int, float]]]
            Dictionary mapping peak_index -> [(candidate_index, score), ...]
            Sorted by score descending, limited to top_k
        total_score : float
            Total score of optimal assignment (only counting assigned peaks)
    """
    # get optimal assignment
    assignment = optimal_assignment(score_matrix, min_score=min_score)
    
    # get top-K candidates for each peak
    top_k_per_peak = {}
    n_peaks = score_matrix.shape[1]
    total_score = 0.0
    
    for peak_idx in range(n_peaks):
        scores_for_peak = score_matrix[:, peak_idx]
        
        top_indices = np.argsort(scores_for_peak)[::-1][:top_k]
        
        # Include all top_k candidates regardless of score (even if 0 or negative)
        # The min_score threshold is only used for optimal assignment, not for potential_compounds
        top_candidates = [
            (int(cand_idx), float(scores_for_peak[cand_idx]))
            for cand_idx in top_indices
        ]
        
        top_k_per_peak[peak_idx] = top_candidates
        
        # Calculate total score for assigned peaks
        if assignment[peak_idx] is not None:
            cand_idx = assignment[peak_idx]
            total_score += float(scores_for_peak[cand_idx])
    
    return assignment, top_k_per_peak, total_score
