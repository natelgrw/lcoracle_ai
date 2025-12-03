import sys
import os
import shutil
import tempfile
import json
import logging
from typing import List, Optional, Dict, Tuple, Union, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to allow importing modules from root
# This adds /app/ (the root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add peak_prophet explicitly to path so its internal absolute imports work
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'peak_prophet'))

# Import modules
# Note: Using try-except blocks to handle potential import errors gracefully
try:
    from amax.pred_lmax import predict_lambda_max
    from retina.pred_rt import predict_retention_time
    from askcos.askcos_scraper import scrape_askcos
    from gradience.pipeline import optimize_gradient, get_compounds
    from peak_prophet.assignment.specsummary import peak_prophesize
    from peak_prophet.predictions.utils.rxn_classes import ChemicalReaction
    from peak_prophet.decoding.LCMS_meas_man import LCMSMeasMan
    from peak_prophet.decoding.LCMSUV_meas_man import LCMSUVMeasMan
except ImportError as e:
    logging.error(f"Failed to import modules: {e}")
    # We'll handle runtime errors in endpoints if modules are missing

app = FastAPI(title="LCOracle.ai API", description="API for LCOracle.ai integrating 5 LC-MS modules")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class AmaxInput(BaseModel):
    compound_smiles: str
    solvent_smiles: str

class AskcosInput(BaseModel):
    reactants: List[str]
    solvent: str

class RetinaInput(BaseModel):
    compound_smiles: str
    solvents: Dict  # {'A': ..., 'B': ...}
    gradient: List[Tuple[float, float]]
    column: Tuple[str, float, float, float]
    flow_rate: float
    temp: float

class GradienceInput(BaseModel):
    reactant_smiles: List[str]
    solvent_smiles: str
    lcms_config: Optional[Dict] = None
    optimizer_config: Optional[Dict] = None

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "LCOracle.ai API is running", "modules": ["amax", "askcos", "retina", "gradience", "peak_prophet"]}

@app.post("/amax/predict")
async def amax_predict(input_data: AmaxInput):
    """Predict UV-Vis absorption maximum."""
    try:
        lmax = predict_lambda_max(input_data.compound_smiles, input_data.solvent_smiles)
        return {"lambda_max": lmax, "unit": "nm"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/askcos/predict")
async def askcos_predict(input_data: AskcosInput):
    """Predict reaction products using ASKCOS scraper."""
    try:
        # scrape_askcos is async
        results = await scrape_askcos(input_data.reactants, input_data.solvent)
        return {"products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retina/predict")
async def retina_predict(input_data: RetinaInput):
    """Predict retention time using ReTiNA."""
    try:
        rt = predict_retention_time(
            compound_smiles=input_data.compound_smiles,
            solvents=input_data.solvents,
            gradient=input_data.gradient,
            column=input_data.column,
            flow_rate=input_data.flow_rate,
            temp=input_data.temp
        )
        return {"retention_time": rt, "unit": "seconds"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gradience/optimize")
async def gradience_optimize(input_data: GradienceInput):
    """Optimize LC-MS gradient."""
    try:
        # get compounds
        compounds = await get_compounds(input_data.reactant_smiles, input_data.solvent_smiles)
        
        if len(compounds) < 2:
            return {"error": "Need at least 2 compounds for optimization", "compounds": compounds}
            
        lcms_config = input_data.lcms_config or {
            'solvents': {
                'A': [{'O': 95.0, 'CO': 5.0}, {'C(=O)O': 0.1}], 
                'B': [{'CC#N': 100.0}, {}] 
            },
            'column': ('RP', 4.6, 150, 5),
            'flow_rate': 1.0, 
            'temp': 40.0,
            'method_length': 15.0
        }
        
        optimizer_config = input_data.optimizer_config or {
            'n_init': 20, # Reduced for web responsiveness
            'max_evals': 50, # Reduced for web responsiveness
            'batch_size': 1,
            'trust_region_init': 0.8,
            'trust_region_min': 0.1,
            'dim': 18,
            'verbose': True
        }
        
        result = optimize_gradient(compounds, lcms_config, optimizer_config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/peakprophet/predict")
async def peakprophet_predict(
    reactants: str = Form(...),
    solvent: str = Form(...),
    ms_file: UploadFile = File(...),
    uv_file: Optional[UploadFile] = File(None),
    method_params: Optional[str] = Form(None)
):
    """
    Predict/Assign peaks from uploaded LC-MS file.
    Reactants and solvent should be JSON strings or simple strings.
    """
    temp_ms_path = None
    temp_uv_path = None
    try:
        # Parse inputs
        # If reactants is a JSON string of list, parse it. If comma separated, split it.
        try:
            reactants_list = json.loads(reactants)
            if not isinstance(reactants_list, list):
                reactants_list = [reactants]
        except:
            reactants_list = [r.strip() for r in reactants.split(",")]
            
        # Save uploaded MS file to temp
        ms_suffix = os.path.splitext(ms_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ms_suffix) as tmp:
            shutil.copyfileobj(ms_file.file, tmp)
            temp_ms_path = tmp.name

        # Save uploaded UV file to temp if present
        if uv_file:
            uv_suffix = os.path.splitext(uv_file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=uv_suffix) as tmp:
                shutil.copyfileobj(uv_file.file, tmp)
                temp_uv_path = tmp.name
            
        # Run PeakProphet logic
        # 1. Create Reaction
        reaction = ChemicalReaction(reactants=reactants_list, solvent=solvent)
        
        # 2. Get Products (Async)
        await reaction.fetch_products_from_askcos()

        # 2.5 Parse method params and run predictions
        if method_params:
            try:
                params = json.loads(method_params)
                
                # Convert lists to tuples where expected
                gradient = [tuple(g) for g in params.get('gradient', [])] if params.get('gradient') else None
                column = tuple(params.get('column')) if params.get('column') else None
                
                reaction.set_lcms_method(
                    solvents=params.get('solvents'),
                    gradient=gradient,
                    column=column,
                    flow_rate=params.get('flow_rate'),
                    temp=params.get('temp')
                )
                
                # Run predictions
                # These methods update the products in-place
                reaction.predict_products_retention_times()
                reaction.predict_products_lambda_max()
                reaction.predict_products_ms_adducts(mode="positive")
                
            except Exception as e:
                logging.error(f"Error setting method params or running predictions: {e}")
                # Continue without predictions if this fails, or raise?
                # Better to log and continue, or maybe the scoring will just be poor.
        
        # 3. Create MeasMan
        # Determine polarity - simple logic or just let it auto-detect
        if temp_uv_path and os.path.exists(temp_uv_path):
             meas_man = LCMSUVMeasMan(mzml_path=temp_ms_path, uvvis_path=temp_uv_path, ms_polarity=1)
        else:
             # Explicitly use positive mode to match the demo and LCMSUV usage
             meas_man = LCMSMeasMan(file_path=temp_ms_path, polarity=1)
        
        # 4. Prophesize
        # Using default weights for now
        spec_summary = peak_prophesize(
            rxn_name=f"Web_Analysis_{ms_file.filename}",
            reaction=reaction,
            meas_man=meas_man,
            output_file=None # Don't save to disk, just return dict
        )
        
        return spec_summary
        
    except Exception as e:
        logging.error(f"PeakProphet Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        if temp_ms_path and os.path.exists(temp_ms_path):
            os.remove(temp_ms_path)
        if temp_uv_path and os.path.exists(temp_uv_path):
            os.remove(temp_uv_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
