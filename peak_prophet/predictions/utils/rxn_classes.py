"""
rxn_classes.py

Author: natelgrw
Last Edited: 12/04/2025

This module provides classes to represent a chemical reaction 
and its predicted products.

It includes a class for predicted products (PredictedProduct)
and a class for chemical reactions (ChemicalReaction).
"""

from typing import List, Optional, Dict, Tuple, Union
import asyncio


try:
    from ..product_pred.askcos_scraper import scrape_askcos
    from ..rt_pred.pred_rt import predict_retention_time_from_smiles
    from ..lmax_pred.pred_lmax import predict_lambda_max, predict_lambda_max_from_tuples
    from ..ms_pred.ms_pred import predict_ms_adducts
except Exception:
    try:
        from product_pred.askcos_scraper import scrape_askcos
        from rt_pred.pred_rt import predict_retention_time_from_smiles
        from lmax_pred.pred_lmax import predict_lambda_max, predict_lambda_max_from_tuples
        from ms_pred.ms_pred import predict_ms_adducts
    except Exception:
        from predictions.product_pred.askcos_scraper import scrape_askcos
        from predictions.rt_pred.pred_rt import predict_retention_time_from_smiles
        from predictions.lmax_pred.pred_lmax import predict_lambda_max, predict_lambda_max_from_tuples
        from predictions.ms_pred.ms_pred import predict_ms_adducts


# ===== Class For Predicted Products ===== #


class PredictedProduct:
    """
    Represents a predicted product of a reaction.
    """

    def __init__(self, smiles: str, probability: float, mol_weight: float, ms_values:  Optional[Dict[str, Tuple[float, float]]] = None, retention_time: Optional[float] = None, lambda_max: Optional[float] = None):
        self.smiles = smiles
        self.probability = probability
        self.mol_weight = mol_weight

        self.ms_values = ms_values

        # retention time in seconds
        self.retention_time = retention_time

        # lambda max in nanometers
        self.lambda_max = lambda_max

    def set_smiles(self, smiles: str):
        """
        Set the SMILES string for the product.
        """
        self.smiles = smiles
    
    def get_smiles(self) -> str:
        """
        Get the SMILES string for the product.
        """
        return self.smiles
    
    def set_ms_values(self, ms_values: Dict[str, Tuple[float, float]]):
        """
        Set MS adduct values. Format: {adduct_name: (mass, probability)}
        """
        self.ms_values = ms_values
    
    def get_ms_values(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get MS adduct values. Format: {adduct_name: (mass, probability)}
        """
        return self.ms_values
    
    def set_probability(self, probability: float):
        """
        Set the probability for the product.
        """
        self.probability = probability
    
    def get_probability(self) -> float:
        """
        Get the probability for the product.
        """
        return self.probability
    
    def set_mol_weight(self, mol_weight: float):
        """
        Set the molecular weight for the product.
        """
        self.mol_weight = mol_weight
    
    def get_mol_weight(self) -> float:
        """
        Get the molecular weight for the product.
        """
        return self.mol_weight

    def set_retention_time(self, retention_time: float):
        """
        Set the retention time for the product.
        """
        self.retention_time = retention_time
    
    def get_retention_time(self) -> Optional[float]:
        """
        Get the retention time for the product.
        """
        return self.retention_time
    
    def set_lambda_max(self, lambda_max: float):
        """
        Set the lambda max for the product.
        """
        self.lambda_max = lambda_max

    def get_lambda_max(self) -> Optional[float]:
        """
        Get the lambda max for the product.
        """
        return self.lambda_max

    def __repr__(self):
        ms_info = f", MS={len(self.ms_values)} adducts" if self.ms_values else ", MS=None"
        rt_info = f"{self.retention_time:.2f}s" if self.retention_time else "None"
        lmax_info = f"{self.lambda_max:.2f}nm" if self.lambda_max else "None"
        return f"PredictedProduct(smiles='{self.smiles}', RT={rt_info}, λmax={lmax_info}{ms_info})"


# ===== Class For Chemical Reactions ===== #


class ChemicalReaction:
    """
    Stores a chemical reaction, its conditions, and predicted products.
    """

    def __init__(self, 
                 reactants: List[str], 
                 solvent: str,
                 lcms_solvents: Optional[Dict] = None,
                 lcms_gradient: Optional[List[Tuple[float, float]]] = None,
                 lcms_column: Optional[Tuple] = None,
                 lcms_flow_rate: Optional[float] = None,
                 lcms_temp: Optional[float] = None):
        """
        Initialize a ChemicalReaction object.
        
        Parameters
        ----------
        reactants : List[str]
            List of reactant SMILES strings
        solvent : str
            Solvent SMILES string (for lambda max prediction)
        lcms_solvents : Dict, optional
            LC-MS solvents encoding. If None, defaults will be used when predicting RT.
            Format: {'A': [{'O': 95.0, 'CO': 5.0}, {}], 'B': [{'CC#N': 100.0}, {}]}
        lcms_gradient : List[Tuple[float, float]], optional
            LC-MS gradient profile. If None, defaults will be used when predicting RT.
            Format: [(time_min, percent_B), ...]
        lcms_column : Tuple, optional
            LC-MS column specification. If None, defaults will be used when predicting RT.
            Format: (type, diameter_mm, length_mm, particle_size_um)
        lcms_flow_rate : float, optional
            LC-MS flow rate in mL/min. If None, defaults will be used when predicting RT.
        lcms_temp : float, optional
            LC-MS temperature in Celsius. If None, defaults will be used when predicting RT.
        """
        self.reactants = reactants
        self.solvent = solvent
        self.products = []
        
        # LC-MS method parameters (None by default, defaults applied when needed)
        self.lcms_solvents = lcms_solvents
        self.lcms_gradient = lcms_gradient
        self.lcms_column = lcms_column
        self.lcms_flow_rate = lcms_flow_rate
        self.lcms_temp = lcms_temp
    
    def set_reactants(self, reactants: List[str]):
        """
        Set the reactants for the reaction.
        """
        self.reactants = reactants
    
    def get_reactants(self) -> List[str]:
        """
        Get the reactants for the reaction.
        """
        return self.reactants
    
    def set_solvent(self, solvent: str):
        """
        Set the solvent for the reaction.
        """
        self.solvent = solvent
    
    def get_solvent(self) -> str:
        """
        Get the solvent for the reaction.
        """
        return self.solvent
    
    def add_product(self, product: PredictedProduct):
        """
        Add a product to the reaction.
        """
        self.products.append(product)
    
    def get_products(self) -> List[PredictedProduct]:
        """
        Get the products for the reaction.
        """
        return self.products
    
    def set_lcms_method(self,
                       solvents: Optional[Dict] = None,
                       gradient: Optional[List[Tuple[float, float]]] = None,
                       column: Optional[Tuple] = None,
                       flow_rate: Optional[float] = None,
                       temp: Optional[float] = None):
        """
        Set LC-MS method parameters.
        
        Parameters
        ----------
        solvents : Dict, optional
            Solvents encoding. Format: {'A': [{'O': 95.0, 'CO': 5.0}, {}], 'B': [{'CC#N': 100.0}, {}]}
        gradient : List[Tuple[float, float]], optional
            Gradient profile. Format: [(time_min, percent_B), ...]
        column : Tuple, optional
            Column specification. Format: (type, diameter_mm, length_mm, particle_size_um)
        flow_rate : float, optional
            Flow rate in mL/min
        temp : float, optional
            Temperature in Celsius
        """
        if solvents is not None:
            self.lcms_solvents = solvents
        if gradient is not None:
            self.lcms_gradient = gradient
        if column is not None:
            self.lcms_column = column
        if flow_rate is not None:
            self.lcms_flow_rate = flow_rate
        if temp is not None:
            self.lcms_temp = temp
    
    def get_lcms_method(self) -> Dict:
        """
        Get current LC-MS method parameters.
        
        Returns
        -------
        Dict
            Dictionary containing all LC-MS method parameters
        """
        return {
            'solvents': self.lcms_solvents,
            'gradient': self.lcms_gradient,
            'column': self.lcms_column,
            'flow_rate': self.lcms_flow_rate,
            'temp': self.lcms_temp
        }

    async def fetch_products_from_askcos(self) -> List[PredictedProduct]:
        """
        Calls the ASKCOS scraper to predict products and populate self.products.
        """
        results = await scrape_askcos(self.reactants, self.solvent)
        self.products = []
        for item in results:
            try:
                smiles = item["smiles"]
                if smiles == "SMILES":
                    continue
                mol_weight = float(item["mol_weight"]) if item["mol_weight"] is not None else None
                probability = float(item["probability"]) if item["probability"] is not None else None
            except (KeyError, ValueError, TypeError):
                continue
            self.add_product(PredictedProduct(smiles=smiles, probability=probability, mol_weight=mol_weight))
        return self.products

    def fetch_products_from_askcos_sync(self) -> List[PredictedProduct]:
        """
        Synchronous wrapper for fetch_products_from_askcos.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.fetch_products_from_askcos())
        raise RuntimeError("fetch_products_from_askcos_sync() cannot run inside an existing event loop. Use the async method instead.")

    def predict_products_retention_times(self, 
                                        solvents: Optional[Dict] = None,
                                        gradient: Optional[List[Tuple[float, float]]] = None,
                                        column: Optional[Tuple] = None,
                                        flow_rate: Optional[float] = None,
                                        temp: Optional[float] = None) -> List[PredictedProduct]:
        """
        Predict retention times for current products and set them in-place.
        
        Uses instance LC-MS method parameters if not provided, but requires
        all parameters to be set (either as arguments or instance variables).
        
        Parameters
        ----------
        solvents : Dict, optional
            Solvents encoding. If None, uses self.lcms_solvents
            Format: {'A': [{'O': 95.0, 'CO': 5.0}, {}], 'B': [{'CC#N': 100.0}, {}]}
        gradient : List[Tuple[float, float]], optional
            Gradient profile. If None, uses self.lcms_gradient
            Format: [(time_min, percent_B), ...]
        column : Tuple, optional
            Column specification. If None, uses self.lcms_column
            Format: (type, diameter_mm, length_mm, particle_size_um)
        flow_rate : float, optional
            Flow rate in mL/min. If None, uses self.lcms_flow_rate
        temp : float, optional
            Temperature in Celsius. If None, uses self.lcms_temp
        
        Returns
        -------
        List[PredictedProduct]
            The updated list of products with retention times in SECONDS.
        
        Raises
        ------
        ValueError
            If any required LC-MS method parameter is not provided and not set as instance variable.
        """
        if not self.products:
            return self.products
        
        # import here to avoid circular imports
        try:
            from ..rt_pred.pred_rt import predict_retention_time_from_list
        except Exception:
            try:
                from rt_pred.pred_rt import predict_retention_time_from_list
            except Exception:
                from predictions.rt_pred.pred_rt import predict_retention_time_from_list
        
        # uses provided parameters or instance variables (no defaults)
        solvents = solvents if solvents is not None else self.lcms_solvents
        gradient = gradient if gradient is not None else self.lcms_gradient
        column = column if column is not None else self.lcms_column
        flow_rate = flow_rate if flow_rate is not None else self.lcms_flow_rate
        temp = temp if temp is not None else self.lcms_temp
        
        # checks that all required parameters are set
        if solvents is None:
            raise ValueError("LC-MS solvents must be provided either as argument or set via set_lcms_method()")
        if gradient is None:
            raise ValueError("LC-MS gradient must be provided either as argument or set via set_lcms_method()")
        if column is None:
            raise ValueError("LC-MS column must be provided either as argument or set via set_lcms_method()")
        if flow_rate is None:
            raise ValueError("LC-MS flow_rate must be provided either as argument or set via set_lcms_method()")
        if temp is None:
            raise ValueError("LC-MS temp must be provided either as argument or set via set_lcms_method()")
        
        # build predictions list with method parameters
        predictions_list = [{
            'compound_smiles': p.get_smiles(),
            'solvents': solvents,
            'gradient': gradient,
            'column': column,
            'flow_rate': flow_rate,
            'temp': temp
        } for p in self.products]
        
        try:
            results_list = predict_retention_time_from_list(predictions_list)
        except Exception as e:
            return self.products
        
        # updating products with retention times
        for product, rt in zip(self.products, results_list):
            if rt is not None:
                try:
                    rt_float = float(rt) if rt is not None else None
                    product.set_retention_time(rt_float)
                except (ValueError, TypeError):
                    continue
        return self.products

    def predict_products_lambda_max(self) -> List[PredictedProduct]:
        """
        Predict lambda max for current products using AMAX_XGB1 model.
        """
        if not self.products:
            return self.products
        
        tuples = [(p.get_smiles(), self.get_solvent()) for p in self.products]
        
        try:
            predictions = predict_lambda_max_from_tuples(tuples)
        except Exception as e:
            return self.products
        
        for product in self.products:
            key = (product.get_smiles(), self.get_solvent())
            if key in predictions:
                try:
                    lm = predictions[key]
                    lm_float = float(lm) if lm is not None else None
                    product.set_lambda_max(lm_float)
                except (ValueError, TypeError):
                    continue
        return self.products

    def predict_products_ms_adducts(self, mode: str = "positive") -> List[PredictedProduct]:
        """
        Predict mass spectrometry adducts for current products.
        
        Calls the MS adduct prediction function and sets ms_values on each product.
        ms_values format: {adduct_name: (mass, probability)}
        """
        if not self.products:
            return self.products
        
        smiles_list = [p.get_smiles() for p in self.products]
        
        try:
            ms_predictions = predict_ms_adducts(smiles_list, mode=mode)
        except Exception as e:
            return self.products
        
        for product in self.products:
            smiles = product.get_smiles()
            if smiles in ms_predictions:
                product.set_ms_values(ms_predictions[smiles])
        
        return self.products

    def __repr__(self):
        lcms_info = f", lcms_method={self.lcms_column[0]}" if self.lcms_column else ""
        return (f"ChemicalReaction(reactants={self.reactants}, solvent='{self.solvent}', "
                f"num_products={len(self.products)}{lcms_info})")
