import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const askcosApi = {
  predict: (reactants: string[], solvent: string) => 
    apiClient.post('/askcos/predict', { reactants, solvent }),
};

export const amaxApi = {
  predict: (compound_smiles: string, solvent_smiles: string) => 
    apiClient.post('/amax/predict', { compound_smiles, solvent_smiles }),
};

export const retinaApi = {
  predict: (data: any) => 
    apiClient.post('/retina/predict', data),
};

export const peakProphetApi = {
  predict: (formData: FormData) => 
    apiClient.post('/peakprophet/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
};

export const gradienceApi = {
  optimize: (data: any) => 
    apiClient.post('/gradience/optimize', data),
};



