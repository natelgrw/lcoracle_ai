#!/usr/bin/env python3
"""
ms_pred.py

Author: natelgrw
Last Edited: 12/04/2025

This script contains functions and information that predicts 
mass spectrometry adducts for a list of SMILES strings.
"""

from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors


# ===== Positive Adducts ===== #


positive_adducts = [{
    "mass": 1.007276,
    "adduct": "[M+3H]3+",
    "multiplicity": 0.33,
}, {
    "mass": 8.334590,
    "adduct": "[M+2H+Na]3+",
    "multiplicity": 0.33,
}, {
    "mass": 15.662453,
    "adduct": "[M+H+2Na]3+",
    "multiplicity": 0.33,
}, {
    "mass": 22.989218,
    "adduct": "[M+3Na]3+",
    "multiplicity": 0.33,
}, {
    "mass": 1.007276,
    "adduct": "[M+2H]2+",
    "multiplicity": 0.5,
}, {
    "mass": 9.520550,
    "adduct": "[M+H+NH4]2+",
    "multiplicity": 0.5,
}, {
    "mass": 11.998247,
    "adduct": "[M+H+Na]2+",
    "multiplicity": 0.5,
}, {
    "mass": 19.985217,
    "adduct": "[M+H+K]2+",
    "multiplicity": 0.5,
}, {
    "mass": 21.520550,
    "adduct": "[M+ACN+2H]2+",
    "multiplicity": 0.5,
}, {
    "mass": 22.989218,
    "adduct": "[M+2Na]2+",
    "multiplicity": 0.5,
}, {
    "mass": 42.033823,
    "adduct": "[M+2ACN+2H]2+",
    "multiplicity": 0.5,
}, {
    "mass": 62.547097,
    "adduct": "[M+3ACN+2H]2+",
    "multiplicity": 0.5,
}, {
    "mass": 1.007276,
    "adduct": "[M+H]+",
    "multiplicity": 1,
}, {
    "mass": 18.033823,
    "adduct": "[M+NH4]+",
    "multiplicity": 1,
}, {
    "mass": 22.989218,
    "adduct": "[M+Na]+",
    "multiplicity": 1,
}, {
    "mass": 33.033489,
    "adduct": "[M+CH3OH+H]+",
    "multiplicity": 1,
}, {
    "mass": 38.963158,
    "adduct": "[M+K]+",
    "multiplicity": 1,
}, {
    "mass": 42.033823,
    "adduct": "[M+ACN+H]+",
    "multiplicity": 1,
}, {
    "mass": 44.971160,
    "adduct": "[M+2Na-H]+",
    "multiplicity": 1,
}, {
    "mass": 61.065340,
    "adduct": "[M+IsoProp+H]+",
    "multiplicity": 1,
}, {
    "mass": 64.015765,
    "adduct": "[M+ACN+Na]+",
    "multiplicity": 1,
}, {
    "mass": 76.919040,
    "adduct": "[M+2K-H]+",
    "multiplicity": 1,
}, {
    "mass": 79.021220,
    "adduct": "[M+DMSO+H]+",
    "multiplicity": 1,
}, {
    "mass": 83.060370,
    "adduct": "[M+2ACN+H]+",
    "multiplicity": 1,
}, {
    "mass": 84.055110,
    "adduct": "[M+IsoProp+Na+H]+",
    "multiplicity": 1,
}, {
    "mass": 1.007276,
    "adduct": "[2M+H]+",
    "multiplicity": 2,
}, {
    "mass": 18.033823,
    "adduct": "[2M+NH4]+",
    "multiplicity": 2,
}, {
    "mass": 22.989218,
    "adduct": "[2M+Na]+",
    "multiplicity": 2,
}, {
    "mass": 38.963158,
    "adduct": "[2M+K]+",
    "multiplicity": 2,
}, {
    "mass": 42.033823,
    "adduct": "[2M+ACN+H]+",
    "multiplicity": 2,
}, {
    "mass": 64.015765,
    "adduct": "[2M+ACN+Na]+",
    "multiplicity": 2,
}]


# ===== Negative Adducts ===== #


negative_adducts = [{
    "mass": -1.007276,
    "multiplicity": 0.333333333,
    "adduct": "[M-3H]3-",
},
    {
    "mass": -1.007276,
    "multiplicity": 0.5,
    "adduct": "[M-2H]2-",
},
    {
    "mass": -19.01839,
    "multiplicity": 1,
    "adduct": "[M-H2O-H]-",
},
    {
    "mass": -1.007276,
    "multiplicity": 1,
    "adduct": "[M-H]-",
},
    {
    "mass": 20.974666,
    "multiplicity": 1,
    "adduct": "[M+Na-2H]-",
},
    {
    "mass": 34.969402,
    "multiplicity": 1,
    "adduct": "[M+Cl]-",
},
    {
    "mass": 36.948606,
    "multiplicity": 1,
    "adduct": "[M+K-2H]-",
},
    {
    "mass": 44.998201,
    "multiplicity": 1,
    "adduct": "[M+FA-H]-",
},
    {
    "mass": 59.013851,
    "multiplicity": 1,
    "adduct": "[M+Hac-H]-",
},
    {
    "mass": 78.918885,
    "multiplicity": 1,
    "adduct": "[M+Br]-",
},
    {
    "mass": 112.985586,
    "multiplicity": 1,
    "adduct": "[M+TFA-H]-",
},
    {
    "mass": -1.007276,
    "multiplicity": 2,
    "adduct": "[2M-H]-",
},
    {
    "mass": 44.998201,
    "multiplicity": 2,
    "adduct": "[2M+FA-H]-",
},
    {
    "mass": 59.013851,
    "multiplicity": 2,
    "adduct": "[2M+Hac-H]-",
},
    {
    "mass": -1.007276,
    "multiplicity": 3,
    "adduct": "[3M-H]-",
}]


# ===== Adduct Propability ===== #


POS_ADDUCT_WEIGHTS = {
    "[M+H]+":        1.00,
    "[M+Na]+":       0.25,
    "[M+NH4]+":      0.20,
    "[M+K]+":        0.15,
    "[M+CH3OH+H]+":  0.08,
    "[M+ACN+H]+":    0.08,
    "[M+ACN+Na]+":   0.06,
    "[M+2H]2+":      0.12,
    "[M+H+Na]2+":    0.03,
    "[2M+H]+":       0.03,
    "[2M+Na]+":      0.015,
    "[M+DMSO+H]+":   0.015,
    "[M+3H]3+":      0.01,
    "default":       0.02,
}

NEG_ADDUCT_WEIGHTS = {
    "[M-H]-":        1.00,
    "[M+Cl]-":       0.22,
    "[M+FA-H]-":     0.16,
    "[M+Hac-H]-":    0.14,
    "[M-H2O-H]-":    0.08,
    "[M+Br]-":       0.04,
    "[2M-H]-":       0.02,
    "[3M-H]-":       0.005,
    "default":       0.02,
}


def normalized_adduct_probs(adduct_list, mode="positive"):
    """
    Return normalized probabilities for adducts of a given mode.
    Each adduct's relative weight is normalized so the total = 1.
    
    Args:
        adduct_list: list of adduct strings (e.g. ["[M+H]+", "[M+Na]+"])
        mode: "positive" or "negative"
    
    Returns:
        dict mapping adduct_name -> normalized probability
    """
    base = POS_ADDUCT_WEIGHTS if mode.lower().startswith("pos") else NEG_ADDUCT_WEIGHTS
    return {a: base.get(a, base["default"]) for a in adduct_list}


# ===== Predicting MS Adducts ===== #


def predict_ms_adducts(smiles_list: List[str], mode: str = "positive") -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Predict expected adduct masses for a list of SMILES strings.

    Args:
        smiles_list: list of SMILES strings
        mode: "positive" or "negative" ionization mode

    Returns:
        {smiles: {adduct_name: (adduct_mass, relative_probability)}}
    """
    results = {}

    adduct_table = positive_adducts if mode.lower().startswith("pos") else negative_adducts
    adduct_names = [a["adduct"] for a in adduct_table]

    probs = normalized_adduct_probs(adduct_names, mode=mode)

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                continue

            mol_weight = Descriptors.ExactMolWt(mol)
            adduct_predictions = {}

            for adduct in adduct_table:
                mass = mol_weight * adduct["multiplicity"] + adduct["mass"]
                name = adduct["adduct"]
                adduct_predictions[name] = (mass, probs.get(name, 0.01))

            results[smiles] = adduct_predictions

        except Exception as e:
            print(f"Error processing {smiles}: {e}")

    return results


# ===== Main ===== #


def main():
    """
    Example usage of the MS adduct predictor.
    """
    test_smiles = [
        "CC(=O)OC(C)=O",
        "CCO",
        "CC(=O)O"
    ]

    mode = "positive"

    print(f"Predicting MS adducts in {mode} mode...\n")
    results = predict_ms_adducts(test_smiles, mode=mode)

    for smiles, adducts in results.items():
        print(f"{smiles}:")
        for name, (mass, prob) in sorted(adducts.items(), key=lambda x: x[1][1], reverse=True):
            print(f"  {name:<20} → {mass:10.4f} Da   (prob={prob:6.3f})")
        print()


if __name__ == "__main__":
    main()