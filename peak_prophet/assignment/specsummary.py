"""
specsummary.py

Author: natelgrw
Last Edited: 11/15/2025

Generates SpecSummary JSON files from ChemicalReaction and measurement manager objects.
Uses score_aggregate.py to score all peaks and assign compounds.
"""

from __future__ import annotations

from typing import Union, Optional, Dict, List, Tuple
import json
import os

from decoding.LCMS_meas_man import LCMSMeasMan
from decoding.LCMSUV_meas_man import LCMSUVMeasMan
from predictions.utils.rxn_classes import ChemicalReaction
from assignment.scoring.score_aggregate import score_all_peaks, optimal_assignment_with_top_k


def peak_prophesize(
    rxn_name: str,
    reaction: ChemicalReaction,
    meas_man: Union[LCMSMeasMan, LCMSUVMeasMan],
    output_file: Optional[str] = None,
    weights: Dict[str, float] = {"rt": 0.2, "uv": 0.2, "ms": 0.4, "prior": 0.2},
    rt_sigma: float = 18.0,
    lmax_sigma: float = 5.0,
    ms_ppm: float = 5.0,
    use_relative_rt: bool = True,
    min_score: float = 0.4,
    top_k: int = 5
) -> Dict:
    """
    Generates a SpecSummary JSON from ChemicalReaction and measurement manager.

    Uses score_aggregate.py to score all compound-peak pairsand assign compounds
    based on the highest scoring compound for each peak.
    
    Parameters
    ----------
    reaction : ChemicalReaction
        ChemicalReaction object with predicted products
    meas_man : LCMSMeasMan or LCMSUVMeasMan
        Measurement manager with detected peaks
    output_file : str, optional
        Path to output JSON file. If None, returns dict without saving.
    weights : Optional[Dict[str, float]]
        Weights for scoring components. Default: {"rt": 0.2, "uv": 0.2, "ms": 0.4, "prior": 0.2}
    rt_sigma : float, default=18.0
        sigma_rt for absolute RT scoring (seconds)
    lmax_sigma : float, default=5.0
        sigma_uv for UV scoring (nm)
    ms_ppm : float, default=5.0
        Parts-per-million tolerance for MS matching
    use_relative_rt : bool, default=True
        If True, use relative order RT scoring
    min_score : float, default=0.4
        Minimum score threshold for assignment
    top_k : int, default=5
        Number of top candidates to include in ranking
    
    Returns
    -------
    Dict
        SpecSummary dictionary (and saves to JSON if output_file provided)
    """
    # determine if UV data is available
    is_uv = isinstance(meas_man, LCMSUVMeasMan)
    
    # get all peaks sorted by RT
    all_peaks = meas_man.get_all_peaks()
    peak_rts = sorted(meas_man.peaks.keys())
    
    if not all_peaks:
        return {
            "contains_UV": is_uv,
            "reaction": "Unknown reaction",
            "reactants": reaction.get_reactants(),
            "solvents": [reaction.get_solvent()],
            "peaks": []
        }
    
    # build score matrix
    score_matrix = score_all_peaks(
        reaction=reaction,
        meas_man=meas_man,
        weights=weights,
        rt_sigma=rt_sigma,
        lmax_sigma=lmax_sigma,
        ms_ppm=ms_ppm,
        use_relative_rt=use_relative_rt
    )
    
    # get optimal assignment and top-K rankings
    assignment, top_k_per_peak, total_score = optimal_assignment_with_top_k(
        score_matrix, top_k=top_k, min_score=min_score
    )
    
    # get products for SMILES lookup
    products = reaction.get_products()
    
    # Warn if no products are available
    if not products:
        print(f"Warning: No predicted products found in reaction. potential_compounds will be empty for all peaks.")
    
    # build peaks list
    peaks_list = []
    for peak_idx, peak_rt in enumerate(peak_rts):
        peak_data = meas_man.peaks[peak_rt]
        
        start_rt = float(peak_data['start_rt'])
        end_rt = float(peak_data['end_rt'])
        apex_rt = float(peak_rt)
        time_range_peak = [start_rt, end_rt]
        
        # get lambda_max if UV data is available
        lmax = None
        if is_uv:
            lmax = peak_data.get('lambda_max')
            if lmax is not None:
                lmax = float(lmax)
        
        # get relative_maxima only if there are actual local maxima (not just apex)
        relative_maxima = None
        local_maxima_rt = peak_data.get('local_maxima_rt', [])
        if local_maxima_rt:
            # only include if there are actual local maxima (not just the apex)
            # relative_maxima stores Absolute Retention Time (X-coordinates)
            relative_maxima = [float(rt) for rt in local_maxima_rt]
        
        assigned_cand_idx = assignment.get(peak_idx)
        optimal_compound = None
        if assigned_cand_idx is not None and assigned_cand_idx < len(products):
            product = products[assigned_cand_idx]
            optimal_compound = product.get_smiles()
        
        potential_compounds = {}
        top_candidates = top_k_per_peak.get(peak_idx, [])
        for cand_idx, score in top_candidates[:top_k]:
            if cand_idx < len(products):
                product = products[cand_idx]
                smiles = product.get_smiles()
                potential_compounds[smiles] = float(score)
            else:
                # this should not happen, but log if it does
                print(f"Warning: candidate index {cand_idx} >= number of products {len(products)} for peak {peak_idx}")
        
        # build peak dictionary
        peak_dict = {
            "time_range": time_range_peak,
            "apex": apex_rt,
            "relative_maxima": relative_maxima,
            "optimal_compound": optimal_compound,
            "potential_compounds": potential_compounds
        }
        
        # Add lmax if available
        if lmax is not None:
            peak_dict["lmax"] = lmax
        
        peaks_list.append(peak_dict)
    
    # build SpecSummary
    spec_summary = {
        "contains_UV": is_uv,
        "reaction": rxn_name,
        "reactants": reaction.get_reactants(),
        "solvents": [reaction.get_solvent()],
    }
    
    # add UV-specific fields
    if is_uv:
        spec_summary["baseline_method"] = "flatfit" 
    
    # calculate min_distance from peak spacing
    if len(peak_rts) > 1:
        min_distances = []
        for i in range(len(peak_rts) - 1):
            min_distances.append(peak_rts[i+1] - peak_rts[i])
        if min_distances:
            spec_summary["min_distance"] = float(min(min_distances))
    else:
        spec_summary["min_distance"] = 0.0
    
    spec_summary["peaks"] = peaks_list
    
    # save to JSON - always save to results directory
    if output_file is None:
        safe_rxn_name = "".join(c for c in rxn_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_rxn_name = safe_rxn_name.replace(' ', '_')
        output_file = f"results/SpecSummary_{safe_rxn_name}.json"
    
    results_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if not os.path.dirname(output_file):
        output_file = os.path.join("results", output_file)
    
    with open(output_file, 'w') as f:
        json.dump(spec_summary, f, indent=2)
    print(f'SpecSummary saved to: {output_file}')
    
    return spec_summary

