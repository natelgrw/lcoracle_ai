#!/usr/bin/env python3
"""
pred_lmax.py

Author: natelgrw
Last Edited: 12/04/2025

This script contains functions to predict UV-Vis absorption maximuma values 
for molecules using AMAX_XGB1. It uses calculate_156_descriptors for both
compound and solvent, concatenating them to create 312 features for the model.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from ..utils.calc_descriptors import calculate_156_descriptors
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from peak_prophet.predictions.utils.calc_descriptors import calculate_156_descriptors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_model_path = None


# ===== Setup Functions ===== #


def _get_model_path():
    """
    Gets the path to the AMAX_XGB1 model file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "AMAX_XGB1", "AMAX_XGB1.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def _load_model():
    """
    Loads the AMAX_XGB1 XGBoost model (lazy loading).
    """
    global _model, _model_path
    if _model is None or _model_path != _get_model_path():
        _model_path = _get_model_path()
        logger.info(f"Loading AMAX_XGB1 model from {_model_path}")
        _model = xgb.Booster()
        _model.load_model(_model_path)
        logger.info("AMAX_XGB1 model loaded successfully")
    return _model


def _get_model_feature_names() -> List[str]:
    """
    Gets the feature names expected by the loaded model.
    
    Returns a list of feature names in the exact order expected by the model.
    """
    model = _load_model()
    if hasattr(model, 'feature_names') and model.feature_names is not None:
        return model.feature_names
    else:
        raise ValueError("Model does not have feature names. Cannot determine feature order.")


def _extract_features_from_descriptors(compound_descriptors: Dict[str, float], 
                                      solvent_descriptors: Dict[str, float],
                                      feature_names: List[str]) -> Optional[np.ndarray]:
    """
    Extracts features from compound and solvent descriptor dictionaries in 
    the order expected by the model.
    """
    if compound_descriptors is None or solvent_descriptors is None:
        return None
    
    features = []
    for feature_name in feature_names:
        if feature_name.endswith("_solv"):
            base_name = feature_name[:-5]
            features.append(solvent_descriptors.get(base_name, 0.0))
        else:
            features.append(compound_descriptors.get(feature_name, 0.0))
    
    features = np.array(features, dtype=np.float32)
    
    if len(features) != 312:
        logger.warning(f"Expected 312 features, got {len(features)}")
        if len(features) < 312:
            features = np.pad(features, (0, 312 - len(features)), 'constant')
        else:
            features = features[:312]
    
    return features


def predict_lambda_max(compound_smiles: str, solvent_smiles: str) -> Optional[float]:
    """
    Predict lambda max for a compound in a given solvent using AMAX_XGB1.
    
    Parameters
    ----------
    compound_smiles : str
        SMILES string of the compound
    solvent_smiles : str
        SMILES string of the solvent
        
    Returns
    -------
    float or None
        Predicted lambda max value, or None if prediction failed
    """
    results = predict_lambda_max_from_tuples([(compound_smiles, solvent_smiles)])
    key = (compound_smiles, solvent_smiles)
    return results.get(key)


def predict_lambda_max_from_tuples(compound_solvent_tuples: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """
    Predict lambda max values for multiple (compound, solvent) pairs using AMAX_XGB1 model.
    
    Parameters
    ----------
    compound_solvent_tuples : List[Tuple[str, str]]
        List of (compound_smiles, solvent_smiles) tuples to predict lambda max for
        
    Returns
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping (compound_smiles, solvent_smiles) tuples to predicted lambda max values
    """
    model = _load_model()
    
    try:
        feature_names = _get_model_feature_names()
    except ValueError:
        if hasattr(model, 'get_score'):
            logger.warning("Could not get feature names from model, using alphabetical order")
            sample_descriptors = calculate_156_descriptors("C")
            if sample_descriptors is None:
                raise ValueError("Could not calculate sample descriptors")
            descriptor_names = sorted(sample_descriptors.keys())
            feature_names = descriptor_names + [name + "_solv" for name in descriptor_names]
        else:
            raise
    
    features_list = []
    valid_tuples = []
    
    for compound_smiles, solvent_smiles in compound_solvent_tuples:
        try:
            compound_descriptors = calculate_156_descriptors(compound_smiles)
        except Exception as e:
            logger.warning(f"Error calculating descriptors for compound SMILES {compound_smiles}: {e}")
            compound_descriptors = None
            
        try:
            solvent_descriptors = calculate_156_descriptors(solvent_smiles)
        except Exception as e:
            logger.warning(f"Error calculating descriptors for solvent SMILES {solvent_smiles}: {e}")
            solvent_descriptors = None
        
        if compound_descriptors is None:
            logger.warning(f"Could not calculate descriptors for compound SMILES: {compound_smiles}")
            continue
            
        if solvent_descriptors is None:
            logger.warning(f"Could not calculate descriptors for solvent SMILES: {solvent_smiles}")
            continue
        
        features = _extract_features_from_descriptors(compound_descriptors, solvent_descriptors, feature_names)
        
        if features is not None:
            features_list.append(features)
            valid_tuples.append((compound_smiles, solvent_smiles))
        else:
            logger.warning(f"Could not extract features for ({compound_smiles}, {solvent_smiles})")
    
    if not features_list:
        logger.error("No valid (compound, solvent) pairs provided")
        return {}
    
    X_df = pd.DataFrame(features_list, columns=feature_names)
    dmatrix = xgb.DMatrix(X_df)
    
    logger.info(f"Making predictions for {len(valid_tuples)} (compound, solvent) pairs")
    predictions = model.predict(dmatrix)
    
    results = {}
    for (compound_smiles, solvent_smiles), pred in zip(valid_tuples, predictions):
        results[(compound_smiles, solvent_smiles)] = float(pred)
    
    return results


# ===== Main ===== #


if __name__ == "__main__":
    test_compounds = ["CCO", "CC(=O)O", "c1ccccc1"]
    test_solvent = "CCO" 
    
    print("Testing lambda max prediction with AMAX_XGB1...")
    print("Compound-Solvent pairs: (CCO, CCO), (CC(=O)O, CCO), (c1ccccc1, CCO)")
    
    tuples = [(compound, test_solvent) for compound in test_compounds]
    results = predict_lambda_max_from_tuples(tuples)
    
    print("Results (using tuples):")
    for (compound, solvent), lmax in results.items():
        print(f"Compound: {compound}, Solvent: {solvent} -> {lmax:.2f} nm")
    
    print("\n" + "="*50 + "\n")
