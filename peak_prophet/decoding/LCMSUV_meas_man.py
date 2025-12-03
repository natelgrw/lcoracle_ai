"""
LCMSUV_meas_man.py

Author: @natelgrw
Last Edited: 11/15/2025

LC-MS/UV-Vis Measurement Manager: Combines LC-MS and UV-Vis data to provide
comprehensive peak information including MS spectra and lambda max values.
Contains functions for easy data processing of chromatograms.
"""

from typing import Dict, List, Tuple, Optional, Literal
from pyopenms import MSExperiment, MzMLFile
from mocca2 import Chromatogram
from mocca2.parsers.wrapper import load_data2d
from mocca2.deconvolution.peak_models import PeakModel
from mocca2 import example_data
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from .LCMS_meas_man import LCMSMeasMan


# ===== LCMSUVMeasMan Class ===== #


class LCMSUVMeasMan:
    """
    Manages combined LC-MS and UV-Vis measurements.
    
    Attributes
    ----------
    peaks : Dict[float, Dict]
        Dictionary mapping apex retention time (float) to peak information.
        Each peak info contains:
        - 'apex_rt': float - Retention time of peak apex
        - 'apex_intensity': float - Intensity at peak apex (UV-Vis)
        - 'start_rt': float - Start boundary retention time
        - 'end_rt': float - End boundary retention time
        - 'local_maxima_rt': List[float] - Retention times of local maxima
        - 'local_maxima_int': List[float] - Intensities of local maxima
        - 'lambda_max': float - Lambda max at peak apex
        - 'ms_spectrum': List[Tuple[float, float]] - MS spectrum at apex as (m/z, intensity) tuples
    mzml_path : str
        Path to the mzML file
    uvvis_path : str
        Path to the UV-Vis file
    chromatogram : Chromatogram
        mocca2 Chromatogram object for UV-Vis data
    ms_meas_man : LCMSMeasMan
        LCMSMeasMan object for MS data
    """
    
    def __init__(self, 
                 mzml_path: str,
                 uvvis_path: Optional[str] = None,
                 use_example_data: bool = False,
                 chromatogram: Optional[Chromatogram] = None,
                 skip_deconvolution: bool = True,
                 deconvolve_algo: PeakModel | Literal['BiGaussian', 'BiGaussianTrailing', 'FraserSuzuki', 'Bemg'] = 'BiGaussian',
                 min_deconvolve_r2: float = 0.95,
                 concentration_relax: bool = False,
                 max_num_peaks: int = 50,
                 contraction_algo: Literal['mean', 'max', 'weighted_mean'] = 'mean',
                 min_h: float = 8.5,
                 min_time: float | None = None,
                 max_time: float | None = None,
                 wavelength: Tuple[int, int] | None = None,
                 method: Literal['asls', 'arpls', 'flatfit'] = "flatfit",
                 time: tuple[int | None, int | None] = None,
                 file_format: Literal['auto', 'empower', 'chemstation', 'labsolutions', 'masslynx'] = 'auto',
                 ms_polarity: Optional[int] = None):
        """
        Initialize LCMSUVMeasMan by loading both MS and UV-Vis data.
        
        Parameters
        ----------
        mzml_path : str
            Path to mzML file for MS data
        uvvis_path : str, optional
            Path to UV-Vis file. If None and use_example_data is False, only MS data is processed.
        use_example_data : bool
            If True, uses mocca2 example_1() data instead of uvvis_path
        chromatogram : Chromatogram, optional
            Pre-loaded mocca2 Chromatogram object. If provided, this takes precedence over uvvis_path and use_example_data.
        skip_deconvolution : bool, default True
            If True, skips the peak deconvolution step (much faster). If False, performs deconvolution.
            Deconvolution is only needed when you have overlapping peaks that need to be separated.
        deconvolve_algo : PeakModel | Literal
            Deconvolution algorithm to use for UV-Vis (only used if skip_deconvolution=False)
        min_deconvolve_r2 : float
            Minimum R² for deconvolution
        concentration_relax : bool
            Whether to relax concentration constraints
        max_num_peaks : int
            Maximum number of peaks to detect
        contraction_algo : Literal
            Contraction algorithm for peak finding
        min_h : float
            Minimum peak height
        min_time : float, optional
            Minimum elution time
        max_time : float, optional
            Maximum elution time
        wavelength : Tuple[int, int], optional
            Wavelength range to extract. Default: (220, 400) nm for typical organic compounds.
        method : Literal
            Baseline correction method (only used if chromatogram is not provided)
        time : tuple, optional
            Time range to extract
        file_format : Literal
            File format specification for UV-Vis
        ms_polarity : int, optional
            MS polarity: 1 for positive, -1 for negative. Auto-detects if None.
        """
        self.mzml_path = mzml_path
        self.uvvis_path = uvvis_path
        
        # initializing MS measurement manager
        print(f'Loading MS data from {mzml_path}...')
        self.ms_meas_man = LCMSMeasMan(mzml_path, polarity=ms_polarity, skip_peak_picking=True)
        
        # loading UV-Vis data
        if chromatogram is not None:
            print('Using provided Chromatogram object for UV-Vis...')
            self.chromatogram = chromatogram
            self.uvvis_path = "provided_chromatogram"
        elif use_example_data:
            print('Using mocca2 example data for UV-Vis...')
            self.chromatogram = example_data.example_1()
            self.uvvis_path = "mocca2_example_data"
        elif uvvis_path is not None:
            print(f'Loading UV-Vis data from {uvvis_path}...')
            # Try multiple encoding approaches for CSV files
            load_successful = False
            last_error = None
            
            # List of encodings to try
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    print(f'Trying encoding: {encoding}...')
                    data2d = load_data2d(uvvis_path, format='auto', encoding=encoding)
                    self.chromatogram = Chromatogram(sample=data2d)
                    # Convert time from minutes to seconds to match MS data
                    self.chromatogram.time = self.chromatogram.time * 60.0
                    print(f'Successfully loaded with {encoding} encoding')
                    print('Converted UV-Vis time axis from minutes to seconds to match MS data')
                    load_successful = True
                    break
                except (UnicodeError, UnicodeDecodeError, ValueError) as e:
                    last_error = e
                    print(f'  {encoding} failed: {type(e).__name__}')
                    continue
            
            if not load_successful:
                raise RuntimeError(
                    f"Failed to load UV-Vis data from '{uvvis_path}' with any encoding. "
                    f"Tried: {', '.join(encodings_to_try)}. "
                    f"Last error: {str(last_error)}. "
                    f"Please check that the file is a valid CSV format."
                ) from last_error
        else:
            raise ValueError("Either uvvis_path must be provided or use_example_data must be True")
        
        # extract wavelength - default to 220-400 nm (typical organic compound range)
        if wavelength is None:
            wavelength = (220, 400)
            print(f'No wavelength range specified, using default: 220-400 nm')
        
        print(f'Extracting wavelength range {wavelength[0]}-{wavelength[1]} nm...')
        self.chromatogram.extract_wavelength(wavelength[0], wavelength[1], inplace=True)
        print(f'Wavelength extraction complete. Data shape: {self.chromatogram.data.shape}')
        
        # extract time if specified
        if time is not None:
            print(f'Extracting time range {time[0]}-{time[1]} min...')
            self.chromatogram.extract_time(time[0], time[1], inplace=True)
            print(f'Time extraction complete')
        
        # correct baseline
        if chromatogram is None:
            print(f'Correcting baseline using {method} method...')
            self.chromatogram.correct_baseline(method)
            print('Baseline correction complete')
        
        # find peaks
        print('Finding peaks...')
        print(f'  Data dimensions: {self.chromatogram.data.shape[0]} wavelengths × {self.chromatogram.data.shape[1]} time points')
        self.chromatogram.find_peaks(
            contraction=contraction_algo, 
            min_height=min_h, 
            min_elution_time=min_time, 
            max_elution_time=max_time
        )
        num_peaks = len(self.chromatogram.peaks) if hasattr(self.chromatogram, 'peaks') else 0
        print(f'Peak finding complete. Found {num_peaks} peaks')
        
        # deconvolve peaks - this can be slow for large datasets
        if skip_deconvolution:
            print('Skipping peak deconvolution (skip_deconvolution=True)')
            print('  If you need deconvolution, set skip_deconvolution=False')
        elif num_peaks > 0:
            print(f'Starting peak deconvolution with {deconvolve_algo} model...')
            print(f'  This may take several minutes for {num_peaks} peaks across {self.chromatogram.data.shape[1]} time points')
            print(f'  Parameters: min_r2={min_deconvolve_r2}, max_comps={max_num_peaks}')
            print('  (This step has no internal progress bar - please be patient)')
            
            import time
            start_time = time.time()
            
            self.chromatogram.deconvolve_peaks(
                model=deconvolve_algo, 
                min_r2=min_deconvolve_r2, 
                relaxe_concs=concentration_relax, 
                max_comps=max_num_peaks
            )
            
            elapsed_time = time.time() - start_time
            print(f'Peak deconvolution complete! Took {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)')
        else:
            print('No peaks found, skipping deconvolution')
        
        # extract 1D signal and time axis for boundary detection
        time_axis = self.chromatogram.time

        # use mean instead of sum to avoid amplifying baseline noise
        signal_1d = self.chromatogram.data.mean(axis=0)
        
        # storing UV-Vis data for plotting (must be done early in case we return early)
        self.uvvis_rt = time_axis
        self.uvvis_intensity = signal_1d
        
        # get peaks - use different methods depending on whether deconvolution was run
        if skip_deconvolution:
            all_peaks = self.chromatogram.peaks if hasattr(self.chromatogram, 'peaks') else []
            
            if not all_peaks:
                self.peaks: Dict[float, Dict] = {}
                print('No UV-Vis peaks found')
                return
            
            # extract peak information
            component_data = []
            for peak in all_peaks:
                apex_idx = peak.maximum  # Index of peak maximum
                if apex_idx < 0 or apex_idx >= len(time_axis):
                    continue
                
                apex_rt = float(time_axis[apex_idx])
                apex_intensity = float(signal_1d[apex_idx])
                
                component_data.append({
                    'apex_rt': apex_rt,
                    'apex_idx': int(apex_idx),
                    'apex_intensity': apex_intensity,
                })
        else:
            # use components from deconvolution
            all_components = list(self.chromatogram.all_components())
            
            if not all_components:
                self.peaks: Dict[float, Dict] = {}
                print('No UV-Vis peaks found')
                return
            
            # extract component information
            component_data = []
            for component in all_components:
                elution_time_idx = component.elution_time
                if elution_time_idx < 0 or elution_time_idx >= len(time_axis):
                    continue
                
                apex_rt = float(time_axis[elution_time_idx])
                apex_intensity = float(signal_1d[elution_time_idx])
                integral = float(component.integral) if hasattr(component, 'integral') else 0.0
                
                component_data.append({
                    'apex_rt': apex_rt,
                    'apex_idx': int(elution_time_idx),
                    'apex_intensity': apex_intensity,
                })
        
        if not component_data:
            self.peaks: Dict[float, Dict] = {}
            print('No valid UV-Vis peaks found')
            return
        
        # sort by retention time
        component_data.sort(key=lambda x: x['apex_rt'])
        
        peak_indices = [comp['apex_idx'] for comp in component_data]
        peak_rt_list = [comp['apex_rt'] for comp in component_data]
        peak_intensity_list = [comp['apex_intensity'] for comp in component_data]
        
        rt_spacing = np.median(np.diff(time_axis))
        min_distance_points = max(1, int(0.2 / rt_spacing))
        valleys, _ = find_peaks(-signal_1d, 
                                distance=max(1, min_distance_points//2),
                                prominence=np.percentile(signal_1d, 20) * 0.3)
        
        self.peaks: Dict[float, Dict] = {}
        
        for i, (peak_idx, peak_rt, peak_intensity) in enumerate(zip(peak_indices, peak_rt_list, peak_intensity_list)):
            start_idx, end_idx, local_maxima = self._find_peak_boundaries_very_strict(
                peak_idx, signal_1d, time_axis, peak_indices, valleys
            )
            
            start_rt = float(time_axis[start_idx])
            end_rt = float(time_axis[end_idx])
            
            local_max_rt = [float(time_axis[idx]) for idx in local_maxima]
            local_max_int = [float(signal_1d[idx]) for idx in local_maxima]
            
            lambda_max = self._get_lambda_max_at_rt(peak_rt)
            
            ms_spectrum = self.ms_meas_man._get_spectrum_at_rt(peak_rt, tolerance=6.0)
            
            # storing peak information
            self.peaks[peak_rt] = {
                'apex_rt': peak_rt,
                'apex_intensity': peak_intensity,
                'start_rt': start_rt,
                'end_rt': end_rt,
                'local_maxima_rt': local_max_rt,
                'local_maxima_int': local_max_int,
                'lambda_max': lambda_max,
                'ms_spectrum': ms_spectrum,
            }
            
            print(f'Peak {i+1}: RT={peak_rt:.3f} min, Boundaries: {start_rt:.3f} - {end_rt:.3f} min, '
                  f'Lambda max: {lambda_max:.1f} nm, Local maxima: {len(local_maxima)}')
    
    def _find_peak_boundaries_very_strict(self, apex_idx, intensity_array, rt_array, all_peaks, all_valleys):
        """
        Find peak boundaries with very strict expansion and maximum limits.
        Improved to handle closely spaced peaks by finding valleys between them.
        
        Parameters
        ----------
        apex_idx : int
            Index of the peak apex
        intensity_array : np.ndarray
            Array of intensity values
        rt_array : np.ndarray
            Array of retention times
        all_peaks : set or list
            Indices of all detected peaks
        all_valleys : set or list
            Indices of all detected valleys
            
        Returns
        -------
        start_idx : int
            Start boundary index
        end_idx : int
            End boundary index
        local_maxima : list
            List of local maxima indices within the peak region
        """
        apex_intensity = intensity_array[apex_idx]
        
        # use 60% drop threshold
        valley_threshold = apex_intensity * 0.65
        
        # don't expand beyond where intensity drops to 30% of apex
        max_expansion_threshold = apex_intensity * 0.3
        
        # limit expansion to reasonable distance from apex
        rt_spacing = np.median(np.diff(rt_array))
        max_expansion_points = int(5.0 / rt_spacing)
        
        # find neighboring peaks
        peaks_list = sorted([p for p in all_peaks if p != apex_idx])
        prev_peak = None
        next_peak = None
        
        for p in peaks_list:
            if p < apex_idx:
                prev_peak = p
            elif p > apex_idx:
                next_peak = p
                break
        
        # find start boundary
        start_idx = apex_idx
        
        # calculate method length for scaling
        total_method_length = rt_array[-1] - rt_array[0]
        
        # peak spacing determination
        if prev_peak is not None:
            peak_spacing = rt_array[apex_idx] - rt_array[prev_peak]
            
            if next_peak is not None:
                peak_spacing_next = rt_array[next_peak] - rt_array[apex_idx]
            else:
                peak_spacing_next = float('inf')

            close_threshold = (16.0 / 120.0) * total_method_length
            moderate_threshold = (32.0 / 120.0) * total_method_length
            
            if peak_spacing < close_threshold:
                max_expansion = (10.0 / 120.0) * total_method_length
            elif peak_spacing < moderate_threshold:
                max_expansion = (16.0 / 120.0) * total_method_length
            elif peak_spacing_next < close_threshold:
                max_expansion = (10.0 / 120.0) * total_method_length
            elif peak_spacing_next < moderate_threshold:
                max_expansion = (16.0 / 120.0) * total_method_length
            else:
                max_expansion = (1.5 / 120.0) * total_method_length
            
            # limit search range to max_expansion from apex
            rt_at_apex = rt_array[apex_idx]
            min_search_rt = rt_at_apex - max_expansion
            search_start = max(0, prev_peak)
            search_end = apex_idx
            
            if search_end > search_start:
                search_indices = [i for i in range(search_start, search_end) if rt_array[i] >= min_search_rt]
                if search_indices:
                    min_idx_in_range = min(search_indices, key=lambda i: intensity_array[i])
                    min_intensity = intensity_array[min_idx_in_range]
                else:
                    min_idx_in_range = search_start
                    min_intensity = intensity_array[min_idx_in_range]
                    for i in range(search_start, search_end):
                        if rt_array[i] >= min_search_rt:
                            min_idx_in_range = i
                            min_intensity = intensity_array[i]
                            break
                

                prev_peak_intensity = intensity_array[prev_peak]
                
                very_close_threshold = (3.0 / 120.0) * total_method_length
                close_threshold_frac = (16.0 / 120.0) * total_method_length
                
                if min_intensity > max(apex_intensity * 0.4, prev_peak_intensity * 0.4):
                    if peak_spacing < very_close_threshold:
                        fraction = 0.9
                    elif peak_spacing < close_threshold_frac:
                        fraction = 0.8
                    else:
                        fraction = 0.7
                    start_idx = prev_peak + int((apex_idx - prev_peak) * fraction)
                else:
                    start_idx = min_idx_in_range
                    
                rt_at_start = rt_array[start_idx]
                rt_distance = rt_at_apex - rt_at_start
                    
                if rt_distance > max_expansion:
                    target_rt = rt_at_apex - max_expansion
                    for i in range(apex_idx - 1, start_idx - 1, -1):
                        if rt_array[i] <= target_rt:
                            start_idx = i
                            break
        else:

            max_expansion = (1.5 / 120.0) * total_method_length
            rt_at_apex = rt_array[apex_idx]
            max_expansion_rt = rt_at_apex - max_expansion
            
            for i in range(apex_idx - 1, -1, -1):
                # stop if exceeded max expansion distance
                if rt_array[i] < max_expansion_rt:
                    start_idx = i + 1
                    break
                
                # stop if intensity drops below 30%
                if intensity_array[i] < max_expansion_threshold:
                    start_idx = i + 1
                    break
                
                # stop if significant valley
                if intensity_array[i] < valley_threshold:
                    if i > 2:
                        avg_ahead = np.mean(intensity_array[max(0, i-2):i+1])
                        if avg_ahead < valley_threshold:
                            start_idx = i + 1
                            break
                
                if i in all_valleys and intensity_array[i] < valley_threshold:
                    start_idx = i
                    break
        
        # find end boundary
        end_idx = apex_idx
        
        if next_peak is not None:
            peak_spacing = rt_array[next_peak] - rt_array[apex_idx]
            
            if prev_peak is not None:
                peak_spacing_prev = rt_array[apex_idx] - rt_array[prev_peak]
            else:
                peak_spacing_prev = float('inf')
            
            close_threshold = (16.0 / 120.0) * total_method_length
            moderate_threshold = (32.0 / 120.0) * total_method_length
            
            if peak_spacing < close_threshold:
                max_expansion = (10.0 / 120.0) * total_method_length
            elif peak_spacing < moderate_threshold:
                max_expansion = (16.0 / 120.0) * total_method_length
            elif peak_spacing_prev < close_threshold:
                max_expansion = (10.0 / 120.0) * total_method_length
            elif peak_spacing_prev < moderate_threshold:
                max_expansion = (16.0 / 120.0) * total_method_length
            else:
                max_expansion = (1.5 / 120.0) * total_method_length
            
            # limit search range to max_expansion from apex
            rt_at_apex = rt_array[apex_idx]
            max_search_rt = rt_at_apex + max_expansion
            search_start = apex_idx
            search_end = min(len(intensity_array), next_peak)
            
            if search_end > search_start:
                search_indices = [i for i in range(search_start, search_end) if rt_array[i] <= max_search_rt]
                if search_indices:
                    min_idx_in_range = min(search_indices, key=lambda i: intensity_array[i])
                    min_intensity = intensity_array[min_idx_in_range]
                else:
                    min_idx_in_range = search_end - 1
                    min_intensity = intensity_array[min_idx_in_range]
                    for i in range(search_end - 1, search_start - 1, -1):
                        if rt_array[i] <= max_search_rt:
                            min_idx_in_range = i
                            min_intensity = intensity_array[i]
                            break

                next_peak_intensity = intensity_array[next_peak]
                
                very_close_threshold = (3.0 / 120.0) * total_method_length
                close_threshold_frac = (16.0 / 120.0) * total_method_length
                
                if min_intensity > max(apex_intensity * 0.4, next_peak_intensity * 0.4):
                    if peak_spacing < very_close_threshold:
                        fraction = 0.9
                    elif peak_spacing < close_threshold_frac:
                        fraction = 0.8
                    else:
                        fraction = 0.7
                    end_idx = apex_idx + int((next_peak - apex_idx) * fraction)
                else:
                    end_idx = min_idx_in_range
                    
                # ensure we don't exceed max_expansion
                rt_at_end = rt_array[end_idx]
                rt_distance = rt_at_end - rt_at_apex
                    
                if rt_distance > max_expansion:
                    target_rt = rt_at_apex + max_expansion
                    for i in range(apex_idx + 1, end_idx + 1):
                        if rt_array[i] >= target_rt:
                            end_idx = i
                            break
        else:
            max_expansion = (1.5 / 120.0) * total_method_length
            rt_at_apex = rt_array[apex_idx]
            max_expansion_rt = rt_at_apex + max_expansion
            
            for i in range(apex_idx + 1, len(intensity_array)):
                # stop if we've exceeded max expansion distance
                if rt_array[i] > max_expansion_rt:
                    end_idx = i - 1
                    break
                
                # stop if intensity drops below 30%
                if intensity_array[i] < max_expansion_threshold:
                    end_idx = i - 1
                    break
                
                # stop if significant valley
                if intensity_array[i] < valley_threshold:
                    if i < len(intensity_array) - 2:
                        avg_ahead = np.mean(intensity_array[i:min(len(intensity_array), i+3)])
                        if avg_ahead < valley_threshold:
                            end_idx = i - 1
                            break
                
                if i in all_valleys and intensity_array[i] < valley_threshold:
                    end_idx = i
                    break
        
        # tighten start if needed
        for i in range(start_idx, apex_idx):
            if intensity_array[i] < max_expansion_threshold:
                start_idx = i + 1
                break
        
        # tighten end if needed
        for i in range(apex_idx, end_idx + 1):
            if intensity_array[i] < max_expansion_threshold:
                end_idx = i - 1
                break
        
        if prev_peak is not None and start_idx <= prev_peak:
            start_idx = prev_peak + 1
        if next_peak is not None and end_idx >= next_peak:
            end_idx = next_peak - 1
        

        min_width_rt = (3.0 / 120.0) * total_method_length
        rt_spacing = np.median(np.diff(rt_array))
        min_width_rt = max(min_width_rt, rt_spacing * 2)
        peak_width = rt_array[end_idx] - rt_array[start_idx]

        close_spacing_threshold = (16.0 / 120.0) * total_method_length
        is_closely_spaced = False
        if prev_peak is not None:
            spacing_to_prev = rt_array[apex_idx] - rt_array[prev_peak]
            if spacing_to_prev < close_spacing_threshold:
                is_closely_spaced = True
        if next_peak is not None:
            spacing_to_next = rt_array[next_peak] - rt_array[apex_idx]
            if spacing_to_next < close_spacing_threshold:
                is_closely_spaced = True
        
        if not is_closely_spaced and peak_width < min_width_rt:
            width_needed = min_width_rt - peak_width
            expand_left = width_needed / 2
            expand_right = width_needed / 2
            
            # expand left
            target_rt_start = rt_array[start_idx] - expand_left
            original_start = start_idx
            for i in range(start_idx - 1, -1, -1):
                if rt_array[i] <= target_rt_start:
                    start_idx = i
                    break
                if prev_peak is not None and i <= prev_peak:
                    start_idx = original_start
                    break
            
            # expand right
            target_rt_end = rt_array[end_idx] + expand_right
            original_end = end_idx
            for i in range(end_idx + 1, len(rt_array)):
                if rt_array[i] >= target_rt_end:
                    end_idx = i
                    break
                if next_peak is not None and i >= next_peak:
                    end_idx = original_end
                    break
            
            if prev_peak is not None and start_idx <= prev_peak:
                start_idx = prev_peak + 1
            if next_peak is not None and end_idx >= next_peak:
                end_idx = next_peak - 1
        
        local_maxima_in_peak = []
        for i in range(start_idx, end_idx + 1):
            if i == apex_idx:
                continue
            if i in all_peaks:
                continue
                
            if i > 0 and i < len(intensity_array) - 1:
                is_local_max = (intensity_array[i-1] < intensity_array[i] and 
                              intensity_array[i+1] < intensity_array[i])
                
                if is_local_max:
                    if intensity_array[i] < apex_intensity * 0.4:
                        continue
                    
                    left_val = intensity_array[i-1] if i > 0 else intensity_array[i]
                    right_val = intensity_array[i+1] if i < len(intensity_array) - 1 else intensity_array[i]
                    prominence = intensity_array[i] - max(left_val, right_val)
                    
                    if prominence < apex_intensity * 0.03:
                        continue
                    
                    window_size = min(2, i - start_idx, end_idx - i)
                    if window_size > 0:
                        window_start = max(start_idx, i - window_size)
                        window_end = min(end_idx, i + window_size)
                        window_avg = np.mean(intensity_array[window_start:window_end+1])
                        if intensity_array[i] < window_avg * 1.05: 
                            continue
                    
                    if len(local_maxima_in_peak) > 0:
                        last_max_idx = local_maxima_in_peak[-1]
                        if abs(i - last_max_idx) < 2:
                            if intensity_array[i] > intensity_array[last_max_idx]:
                                local_maxima_in_peak[-1] = i
                            continue
                    
                    local_maxima_in_peak.append(i)
        
        return start_idx, end_idx, local_maxima_in_peak
    
    def _get_lambda_max_at_rt(self, target_rt: float, tolerance: float = 0.1) -> Optional[float]:
        """
        Extract lambda max at a specific retention time.
        
        Parameters
        ----------
        target_rt : float
            Target retention time (in minutes)
        tolerance : float
            Maximum time difference to search for spectrum (in minutes)
            
        Returns
        -------
        float or None
            Lambda max value in nm, or None if not found
        """
        # find time index closest to target_rt
        time_axis = self.chromatogram.time
        closest_idx = int(np.argmin(np.abs(time_axis - target_rt)))
        closest_rt = float(time_axis[closest_idx])
        
        if abs(closest_rt - target_rt) > tolerance:
            return None

        spectrum = self.chromatogram.data[:, closest_idx]
        
        # get wavelength axis
        wavelengths = getattr(self.chromatogram, 'wavelengths', None)
        if wavelengths is None:
            wavelengths = getattr(self.chromatogram, 'wavelength', None)
        if wavelengths is None:
            return None
        
        # find wavelength with maximum absorbance
        max_idx = int(np.argmax(spectrum))
        lambda_max = float(wavelengths[max_idx]) if hasattr(wavelengths, '__len__') else float(wavelengths)
        
        return lambda_max
    
    def get_peak(self, apex_rt: float) -> Optional[Dict]:
        """
        Get peak information by apex retention time.
        
        Parameters
        ----------
        apex_rt : float
            Retention time of peak apex
            
        Returns
        -------
        Dict or None
            Peak information dictionary, or None if not found
        """
        return self.peaks.get(apex_rt)
    
    def get_all_peaks(self) -> List[Dict]:
        """
        Get all peaks as a list, sorted by apex retention time.
        
        Returns
        -------
        List[Dict]
            List of peak information dictionaries
        """
        return [self.peaks[rt] for rt in sorted(self.peaks.keys())]
    
    def get_apex_times(self) -> List[float]:
        """
        Get all apex retention times, sorted.
        
        Returns
        -------
        List[float]
            Sorted list of apex retention times
        """
        return sorted(self.peaks.keys())
    
    def get_lambda_max(self, apex_rt: float) -> Optional[float]:
        """
        Get lambda max for a specific peak by apex retention time.
        
        Parameters
        ----------
        apex_rt : float
            Retention time of peak apex
            
        Returns
        -------
        float or None
            Lambda max value in nm, or None if peak not found
        """
        peak = self.peaks.get(apex_rt)
        if peak is None:
            return None
        return peak.get('lambda_max')
    
    def get_ms_spectrum(self, apex_rt: float) -> Optional[List[Tuple[float, float]]]:
        """
        Get MS spectrum for a specific peak by apex retention time.
        
        Parameters
        ----------
        apex_rt : float
            Retention time of peak apex
            
        Returns
        -------
        List[Tuple[float, float]] or None
            List of (m/z, intensity) tuples, or None if peak not found
        """
        peak = self.peaks.get(apex_rt)
        if peak is None:
            return None
        return peak.get('ms_spectrum', [])
    
    def plot_uvvis_with_peaks(self, output_file: Optional[str] = None, show: bool = True):
        """
        Plot the UV-Vis chromatogram with peaks, boundaries, and local maxima.
        
        Parameters
        ----------
        output_file : str, optional
            Output file path for the plot. If None, uses input filename with suffix.
        show : bool
            Whether to display the plot using plt.show()
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.uvvis_rt, self.uvvis_intensity, color='blue', linewidth=0.8, 
                alpha=0.7, label='UV-Vis Chromatogram')
        
        all_peaks = self.get_all_peaks()
        for i, peak_info in enumerate(all_peaks):
            ax.axvspan(peak_info['start_rt'], peak_info['end_rt'], 
                       alpha=0.2, color='orange', zorder=1)
            ax.axvline(peak_info['start_rt'], color='orange', linestyle='--', 
                       linewidth=1, alpha=0.6, zorder=2)
            ax.axvline(peak_info['end_rt'], color='orange', linestyle='--', 
                       linewidth=1, alpha=0.6, zorder=2)
            
            if len(peak_info['local_maxima_rt']) > 0:
                ax.scatter(peak_info['local_maxima_rt'], peak_info['local_maxima_int'],
                          color='purple', s=100, marker='s', alpha=0.7, zorder=4,
                          edgecolors='darkviolet', linewidths=1.5, label='Local maxima' if i == 0 else '')
        
        # plot peak apexes
        apex_rt_list = [peak['apex_rt'] for peak in all_peaks]
        apex_int_list = [peak['apex_intensity'] for peak in all_peaks]
        
        ax.scatter(apex_rt_list, apex_int_list, color='red', s=200, marker='v', 
                  zorder=5, label=f'Peak Apex ({len(apex_rt_list)})', 
                  edgecolors='darkred', linewidths=2)
        
        # annotate peak numbers with lambda max
        for i, (rt, intensity) in enumerate(zip(apex_rt_list, apex_int_list)):
            peak_info = all_peaks[i]
            lambda_max = peak_info.get('lambda_max')
            label = f'{i+1}'
            if lambda_max is not None:
                label += f'\nλ={lambda_max:.0f}nm'
            ax.annotate(label, xy=(rt, intensity), xytext=(5, 10),
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        color='darkred', zorder=6)
        
        ax.set_xlabel('Retention Time (min)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)
        filename = os.path.basename(self.uvvis_path) if self.uvvis_path else "UV-Vis"
        ax.set_title(f'{filename} - Peak Picking with Lambda Max', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        if output_file is not None:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f'Plot saved to: {output_file}')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_ms_spectrum(self, target_rt: float, tolerance: float = 6.0, 
                        output_file: Optional[str] = None, show: bool = True,
                        top_n: Optional[int] = None):
        """
        Plot MS spectrum at a specific retention time.
        
        Parameters
        ----------
        target_rt : float
            Target retention time (in minutes)
        tolerance : float
            Maximum time difference to search for spectrum (in seconds)
        output_file : str, optional
            Output file path for the plot. If None, uses input filename with suffix.
        show : bool
            Whether to display the plot using plt.show()
        top_n : int, optional
            If specified, only plot the top N most intense peaks
        """
        # use the MS measurement manager to plot the spectrum
        self.ms_meas_man.plot_ms_spectrum(target_rt, tolerance=tolerance, 
                                          output_file=output_file, show=show, top_n=top_n)

    def __len__(self) -> int:
        """Return the number of peaks in the collection."""
        return len(self.peaks)

    def __repr__(self) -> str:
        """String representation of the LCMSUVMeasMan."""
        return f"LCMSUVMeasMan(mzml_path='{self.mzml_path}', uvvis_path='{self.uvvis_path}', num_peaks={len(self.peaks)})"
