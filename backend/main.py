"""
main.py

Author: natelgrw
Last Edited: 12/04/2025

File containing the backend FastAPI application for LCOracle.ai.
"""

import sys
import os
import shutil
import tempfile
import json
import logging
from typing import List, Optional, Dict, Tuple, Union, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'peak_prophet'))

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

# initialize rate limiter
enable_rate_limits = os.getenv("ENABLE_RATE_LIMITS", "False").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=enable_rate_limits)
app = FastAPI(
    title="LCOracle.ai API",
    description="API for LCOracle.ai integrating 5 LC-MS modules",
    root_path=os.getenv("ROOT_PATH", "")
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# security to restrict CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://lcoracle-ai.vercel.app",
    "https://lcoracle.ai",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Model Inputs ===== #


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


# ===== API Endpoints ===== #


@app.get("/")
async def root():
    return {"message": "LCOracle.ai API is running", "modules": ["amax", "askcos", "retina", "gradience", "peak_prophet"]}


@app.post("/amax/predict")
@limiter.limit("20/minute")
async def amax_predict(request: Request, input_data: AmaxInput):
    """
    Predicts UV-Vis absorption maxima for a compound-solvent combination.
    """
    try:
        lmax = predict_lambda_max(input_data.compound_smiles, input_data.solvent_smiles)
        return {"lambda_max": lmax, "unit": "nm"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/askcos/predict")
@limiter.limit("10/minute")
async def askcos_predict(request: Request, input_data: AskcosInput):
    """
    Predicts reaction products using an ASKCOS scraper.
    """
    try:
        results = await scrape_askcos(input_data.reactants, input_data.solvent)
        return {"products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retina/predict")
@limiter.limit("20/minute")
async def retina_predict(request: Request, input_data: RetinaInput):
    """
    Predicts retention time for a compound using ReTiNA.
    """
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
@limiter.limit("5/day")
async def gradience_optimize(request: Request, input_data: GradienceInput):
    """
    Utilizer the Gradience module to optimize LC-MS gradient.
    """
    try:
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
            'n_init': 20,
            'max_evals': 50,
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
@limiter.limit("5/day")
async def peakprophet_predict(
    request: Request,
    reactants: str = Form(...),
    solvent: str = Form(...),
    ms_file: UploadFile = File(...),
    uv_file: Optional[UploadFile] = File(None),
    method_params: Optional[str] = Form(None),
):
    """
    Predicts and assigns peaks from uploaded LC-MS file.
    Reactants and solvent should be JSON strings or simple strings.
    """

    temp_ms_path = None
    temp_uv_path = None
    try:
        try:
            reactants_list = json.loads(reactants)
            if not isinstance(reactants_list, list):
                reactants_list = [reactants]
        except:
            reactants_list = [r.strip() for r in reactants.split(",")]
            
        # save uploaded MS file to temp file
        ms_suffix = os.path.splitext(ms_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ms_suffix) as tmp:
            shutil.copyfileobj(ms_file.file, tmp)
            temp_ms_path = tmp.name

        # save uploaded UV file to temp file
        if uv_file:
            uv_suffix = os.path.splitext(uv_file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=uv_suffix) as tmp:
                shutil.copyfileobj(uv_file.file, tmp)
                temp_uv_path = tmp.name
            
        # run peakprophet logic
        reaction = ChemicalReaction(reactants=reactants_list, solvent=solvent)
        
        await reaction.fetch_products_from_askcos()

        if method_params:
            try:
                params = json.loads(method_params)
                
                gradient = [tuple(g) for g in params.get('gradient', [])] if params.get('gradient') else None
                column = tuple(params.get('column')) if params.get('column') else None
                
                reaction.set_lcms_method(
                    solvents=params.get('solvents'),
                    gradient=gradient,
                    column=column,
                    flow_rate=params.get('flow_rate'),
                    temp=params.get('temp')
                )

                reaction.predict_products_retention_times()
                reaction.predict_products_lambda_max()
                reaction.predict_products_ms_adducts(mode="positive")
                
            except Exception as e:
                logging.error(f"Error setting method params or running predictions: {e}")

        if temp_uv_path and os.path.exists(temp_uv_path):
             meas_man = LCMSUVMeasMan(mzml_path=temp_ms_path, uvvis_path=temp_uv_path, ms_polarity=1)
        else:
             meas_man = LCMSMeasMan(file_path=temp_ms_path, polarity=1)

        spec_summary = peak_prophesize(
            rxn_name=f"Web_Analysis_{ms_file.filename}",
            reaction=reaction,
            meas_man=meas_man,
            output_file=None
        )
        
        return spec_summary
        
    except Exception as e:
        logging.error(f"PeakProphet Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_ms_path and os.path.exists(temp_ms_path):
            os.remove(temp_ms_path)
        if temp_uv_path and os.path.exists(temp_uv_path):
            os.remove(temp_uv_path)


# ===== Main Function =====


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
