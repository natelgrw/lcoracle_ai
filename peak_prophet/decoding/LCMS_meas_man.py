"""
LCMS_meas_man.py

Author: @natelgrw
Last Edited: 12/04/2025

LC-MS Measurement Manager: Stores peak information including apexes, boundaries, 
local maxima, and MS spectra for each peak. Contains functions for easy data processing
of chromatograms.
"""

from typing import Dict, List, Tuple, Optional
from pyopenms import MSExperiment, MzMLFile, MzXMLFile
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os


# ===== LCMSMeasMan Class ===== #


class LCMSMeasMan:
    """
    Manages LC-MS measurements including peak detection and MS spectra extraction.
    
    Attributes
    ----------
    peaks : Dict[float, Dict]
        Dictionary mapping apex retention time (float) to peak information.
        Each peak info contains:
        - 'apex_rt': float - Retention time of peak apex
        - 'apex_intensity': float - Intensity at peak apex
        - 'start_rt': float - Start boundary retention time
        - 'end_rt': float - End boundary retention time
        - 'local_maxima_rt': List[float] - Retention times of local maxima
        - 'local_maxima_int': List[float] - Intensities of local maxima
        - 'ms_spectrum': List[Tuple[float, float]] - MS spectrum at apex as (m/z, intensity) tuples
    file_path : str
        Path to the source file
    exp : MSExperiment
        PyOpenMS experiment object (loaded once for spectrum extraction)
    """
    
    def __init__(self, file_path: str, polarity: Optional[int] = None, skip_peak_picking: bool = False):
        """
        Initialize LCMSMeasMan by loading file and performing peak picking.
        
        Parameters
        ----------
        file_path : str
            Path to mzML or mzXML file
        polarity : int, optional
            Explicitly set polarity: 1 for positive, -1 for negative.
            If None, auto-detects: uses positive if available, otherwise negative.
        skip_peak_picking : bool, default False
            If True, skip peak picking on TIC. Only loads MS data and sets up polarity.
            Useful when peaks are detected from another source (e.g., UV-Vis).
        """
        self.file_path = file_path
        
        # load MSExperiment - detect file type and use appropriate loader
        print(f'Loading {file_path}...')
        self.exp = MSExperiment()
        
        # Detect file type by extension
        file_lower = file_path.lower()
        if file_lower.endswith('.mzxml'):
            print('Detected mzXML format')
            MzXMLFile().load(file_path, self.exp)
        elif file_lower.endswith('.mzml'):
            print('Detected mzML format')
            MzMLFile().load(file_path, self.exp)
        else:
            # Default to MzMLFile for backward compatibility
            print('File extension not recognized, attempting mzML format')
            MzMLFile().load(file_path, self.exp)
        
        # get TIC
        tic = self.exp.calculateTIC()
        rt_array, intensity_array = tic.get_peaks()
        
        # baseline correction
        min_intensity = float(intensity_array.min())
        corrected_intensity = intensity_array - min_intensity
        
        # detect available polarities
        positive_rt_set = set()
        negative_rt_set = set()
        for spec in self.exp:
            rt = spec.getRT()
            spec_polarity = spec.getInstrumentSettings().getPolarity()
            closest_idx = np.argmin(np.abs(rt_array - rt))
            closest_rt = rt_array[closest_idx]
            if spec_polarity == 1:
                positive_rt_set.add(closest_rt)
            elif spec_polarity == -1:
                negative_rt_set.add(closest_rt)
        
        if polarity is None:
            if len(positive_rt_set) > 0:
                polarity = 1
                print(f'Auto-selected: MS+ (positive mode) - {len(positive_rt_set)} points found')
            elif len(negative_rt_set) > 0:
                polarity = -1 
                print(f'Auto-selected: MS- (negative mode) - {len(negative_rt_set)} points found')
            else:
                raise ValueError("No MS+ or MS- data found in file")
        else:
            if polarity == 1:
                print(f'Using: MS+ (positive mode) - {len(positive_rt_set)} points found')
            elif polarity == -1:
                print(f'Using: MS- (negative mode) - {len(negative_rt_set)} points found')
            else:
                raise ValueError(f"Invalid polarity: {polarity}. Use 1 for positive or -1 for negative")
        
        if polarity == 1:
            selected_rt_set = positive_rt_set
            polarity_label = "MS+"
        else:
            selected_rt_set = negative_rt_set
            polarity_label = "MS-"
        
        if len(selected_rt_set) == 0:
            raise ValueError(f"No {polarity_label} data found in file")
        
        # create filtered chromatogram
        selected_mask = np.array([rt in selected_rt_set for rt in rt_array])
        selected_rt = rt_array[selected_mask]
        selected_intensity = corrected_intensity[selected_mask]
        
        # sort by RT
        sort_idx = np.argsort(selected_rt)
        selected_rt_sorted = selected_rt[sort_idx]
        selected_intensity_sorted = selected_intensity[sort_idx]
        
        print(f'{polarity_label} only data: {len(selected_rt_sorted)} points')
        print(f'RT range: {selected_rt_sorted[0]:.2f} - {selected_rt_sorted[-1]:.2f} s ({selected_rt_sorted[0]/60:.2f} - {selected_rt_sorted[-1]/60:.2f} min)')
        
        self.tic_rt = selected_rt_sorted
        self.tic_intensity = selected_intensity_sorted
        self.polarity_label = polarity_label
        self.selected_polarity = polarity
        self.peaks: Dict[float, Dict] = {}
        
        if skip_peak_picking:
            print('Skipping TIC peak picking (peaks will be provided from external source)')
            return
        
        # peak picking
        height_threshold = np.percentile(selected_intensity_sorted, 27.5)
        rt_spacing = np.median(np.diff(selected_rt_sorted))
        min_distance_points = max(1, int(0.2 / rt_spacing))
        prominence_threshold = np.percentile(selected_intensity_sorted, 20) * 0.5
        
        peaks, properties = find_peaks(selected_intensity_sorted, 
                                       height=height_threshold,
                                       distance=min_distance_points,
                                       prominence=prominence_threshold)
        
        rt_picked = selected_rt_sorted[peaks]
        int_picked = selected_intensity_sorted[peaks]
        
        print(f'Found {len(rt_picked)} peak apexes')
        
        valleys, _ = find_peaks(-selected_intensity_sorted, 
                                distance=max(1, min_distance_points//2),
                                prominence=np.percentile(selected_intensity_sorted, 20) * 0.3)
        
        for i, peak_idx in enumerate(peaks):
            peak_rt = float(rt_picked[i])
            peak_intensity = float(int_picked[i])
            
            start_idx, end_idx, local_maxima = self._find_peak_boundaries_very_strict(
                peak_idx, selected_intensity_sorted, selected_rt_sorted, peaks, valleys
            )
            
            start_rt = float(selected_rt_sorted[start_idx])
            end_rt = float(selected_rt_sorted[end_idx])
            
            local_max_rt = [float(selected_rt_sorted[idx]) for idx in local_maxima]
            local_max_int = [float(selected_intensity_sorted[idx]) for idx in local_maxima]
            
            ms_spectrum = self._get_spectrum_at_rt(peak_rt)
            
            self.peaks[peak_rt] = {
                'apex_rt': peak_rt,
                'apex_intensity': peak_intensity,
                'start_rt': start_rt,
                'end_rt': end_rt,
                'local_maxima_rt': local_max_rt,
                'local_maxima_int': local_max_int,
                'ms_spectrum': ms_spectrum
            }
            
            print(f'Peak {i+1}: RT={peak_rt:.3f} s, Boundaries: {start_rt:.3f} - {end_rt:.3f} s, Local maxima: {len(local_maxima)}')
    
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
        
        valley_threshold = apex_intensity * 0.65
        
        max_expansion_threshold = apex_intensity * 0.3
        
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
        
        # calculate method length for scaling
        total_method_length = rt_array[-1] - rt_array[0]
        
        # find start boundary
        start_idx = apex_idx
        
        if prev_peak is not None:
            search_start = max(0, prev_peak)
            search_end = apex_idx
            if search_end > search_start:
                min_idx = search_start + np.argmin(intensity_array[search_start:search_end])
                min_intensity = intensity_array[min_idx]
                

                prev_peak_intensity = intensity_array[prev_peak]
                if min_intensity > max(apex_intensity * 0.5, prev_peak_intensity * 0.5):
                    start_idx = (prev_peak + apex_idx) // 2
                else:
                    start_idx = min_idx
                    
                # don't expand more than 3/120 * method length from apex
                rt_at_start = rt_array[start_idx]
                rt_at_apex = rt_array[apex_idx]
                max_expansion = (3.0 / 120.0) * total_method_length
                if rt_at_apex - rt_at_start > max_expansion:
                    target_rt = rt_at_apex - max_expansion
                    for i in range(apex_idx - 1, start_idx - 1, -1):
                        if rt_array[i] <= target_rt:
                            start_idx = i
                            break
        else:
            for i in range(apex_idx - 1, max(0, apex_idx - max_expansion_points) - 1, -1):
                if intensity_array[i] < max_expansion_threshold:
                    start_idx = i + 1
                    break
                
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
            search_start = apex_idx
            search_end = min(len(intensity_array), next_peak)
            if search_end > search_start:
                min_idx = search_start + np.argmin(intensity_array[search_start:search_end])
                min_intensity = intensity_array[min_idx]

                next_peak_intensity = intensity_array[next_peak]
                if min_intensity > max(apex_intensity * 0.5, next_peak_intensity * 0.5):
                    end_idx = (apex_idx + next_peak) // 2
                else:
                    end_idx = min_idx
                    
                # don't expand more than 3/120 * method length from apex
                rt_at_end = rt_array[end_idx]
                rt_at_apex = rt_array[apex_idx]
                max_expansion = (3.0 / 120.0) * total_method_length
                if rt_at_end - rt_at_apex > max_expansion:
                    target_rt = rt_at_apex + max_expansion
                    for i in range(apex_idx + 1, end_idx + 1):
                        if rt_array[i] >= target_rt:
                            end_idx = i
                            break
        else:
            for i in range(apex_idx + 1, min(len(intensity_array), apex_idx + max_expansion_points) + 1):
                if intensity_array[i] < max_expansion_threshold:
                    end_idx = i - 1
                    break
                
                if intensity_array[i] < valley_threshold:
                    if i < len(intensity_array) - 2:
                        avg_ahead = np.mean(intensity_array[i:min(len(intensity_array), i+3)])
                        if avg_ahead < valley_threshold:
                            end_idx = i - 1
                            break
                
                if i in all_valleys and intensity_array[i] < valley_threshold:
                    end_idx = i
                    break
        
        # ensure boundaries don't exceed intensity threshold
        for i in range(start_idx, apex_idx):
            if intensity_array[i] < max_expansion_threshold:
                start_idx = i + 1
                break
        
        for i in range(apex_idx, end_idx + 1):
            if intensity_array[i] < max_expansion_threshold:
                end_idx = i - 1
                break
        
        # ensure boundaries don't overlap with neighboring peaks
        if prev_peak is not None and start_idx <= prev_peak:
            start_idx = prev_peak + 1
        if next_peak is not None and end_idx >= next_peak:
            end_idx = next_peak - 1
        
        # ensure minimum peak width: 3/120 (2.5%) of total method length
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
            
            target_rt_start = rt_array[start_idx] - expand_left
            original_start = start_idx

            # expand left side
            for i in range(start_idx - 1, -1, -1):
                if rt_array[i] <= target_rt_start:
                    start_idx = i
                    break
                if prev_peak is not None and i <= prev_peak:
                    start_idx = original_start
                    break
            
            # expand right side
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
                    
                    if prominence < apex_intensity * 0.03:  # Was 0.05
                        continue
                    
                    window_size = min(2, i - start_idx, end_idx - i)
                    if window_size > 0:
                        window_start = max(start_idx, i - window_size)
                        window_end = min(end_idx, i + window_size)
                        window_avg = np.mean(intensity_array[window_start:window_end+1])
                        if intensity_array[i] < window_avg * 1.05:  # Was 1.10
                            continue
                    
                    if len(local_maxima_in_peak) > 0:
                        last_max_idx = local_maxima_in_peak[-1]
                        if abs(i - last_max_idx) < 2:
                            if intensity_array[i] > intensity_array[last_max_idx]:
                                local_maxima_in_peak[-1] = i
                            continue
                    
                    local_maxima_in_peak.append(i)
        
        return start_idx, end_idx, local_maxima_in_peak
    
    def _get_spectrum_at_rt(self, target_rt: float, tolerance: float = 6.0) -> List[Tuple[float, float]]:
        """
        Extracts MS spectrum at a specific retention time.
        Always returns the nearest spectrum with matching polarity, regardless of tolerance.
        
        Parameters
        ----------
        target_rt : float
            Target retention time (in seconds)
        tolerance : float
            Maximum time difference to warn about (in seconds). Not used for filtering.
            
        Returns
        -------
        List[Tuple[float, float]]
            List of (m/z, intensity) tuples. Empty list if no spectrum found.
        """
        # finds the spectrum closest to target_rt with matching polarity
        best_spec = None
        best_rt_diff = float('inf')
        
        for spec in self.exp:
            spec_rt = spec.getRT()
            spec_polarity = spec.getInstrumentSettings().getPolarity()
            
            # only consider spectra with matching polarity
            if spec_polarity != self.selected_polarity:
                continue
            
            rt_diff = abs(spec_rt - target_rt)
            
            # find the closest spectrum
            if rt_diff < best_rt_diff:
                best_spec = spec
                best_rt_diff = rt_diff
        
        if best_spec is None:
            return []
        
        if best_rt_diff > tolerance:
            print(f'Warning: Closest MS spectrum at RT = {target_rt:.3f} s is {best_rt_diff:.2f} s away.')
        
        mz_array, intensity_array = best_spec.get_peaks()
        
        spectrum = [(float(mz), float(intensity)) for mz, intensity in zip(mz_array, intensity_array)]
        
        return spectrum
    
    def get_peak(self, apex_rt: float) -> Optional[Dict]:
        """
        Gets peak information by apex retention time.
        
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
        Gets all peaks as a list, sorted by apex retention time.
        
        Returns
        -------
        List[Dict]
            List of peak information dictionaries
        """
        return [self.peaks[rt] for rt in sorted(self.peaks.keys())]
    
    def get_apex_times(self) -> List[float]:
        """
        Gets all apex retention times, sorted.
        
        Returns
        -------
        List[float]
            Sorted list of apex retention times
        """
        return sorted(self.peaks.keys())
    
    def get_ms_spectrum(self, apex_rt: float) -> Optional[List[Tuple[float, float]]]:
        """
        Gets MS spectrum for a specific peak by apex retention time.
        
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
    
    def __len__(self) -> int:
        """Returns the number of peaks in the collection."""
        return len(self.peaks)
    
    def plot_ms_spectrum(self, target_rt: float, tolerance: float = 30.0, 
                        output_file: Optional[str] = None, show: bool = True,
                        top_n: Optional[int] = None):
        """
        Plots MS spectrum at a specific retention time.
        
        Parameters
        ----------
        target_rt : float
            Target retention time (in seconds)
        tolerance : float
            Maximum time difference to search for spectrum (in seconds). Default is 30 seconds.
        output_file : str, optional
            Output file path for the plot. If None, uses input filename with suffix.
        show : bool
            Whether to display the plot using plt.show()
        top_n : int, optional
            If specified, only plot the top N most intense peaks
        """
        # get target spectrum
        spectrum = self._get_spectrum_at_rt(target_rt, tolerance=tolerance)
        
        if not spectrum:
            print(f'No MS spectrum found at RT = {target_rt:.3f} s (within {tolerance} s tolerance)')
            return
        
        # sort by intensity and optionally filter to top N
        sorted_spectrum = sorted(spectrum, key=lambda x: x[1], reverse=True)
        if top_n is not None:
            sorted_spectrum = sorted_spectrum[:top_n]
        
        mz_values = [x[0] for x in sorted_spectrum]
        intensity_values = [x[1] for x in sorted_spectrum]
        
        # plot
        fig, ax = plt.subplots(figsize=(12, 6))
        markerline, stemlines, baseline = ax.stem(mz_values, intensity_values, basefmt=' ', linefmt='b-', markerfmt='bo')
        markerline.set_markersize(4)
        
        ax.set_xlabel('m/z', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        filename = os.path.basename(self.file_path)
        ax.set_title(f'{filename} - MS Spectrum at RT = {target_rt:.3f} s ({target_rt/60:.2f} min) ({self.polarity_label})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # annotate top peaks
        if len(sorted_spectrum) > 0:
            top_peaks = sorted_spectrum[:min(10, len(sorted_spectrum))]
            for mz, intensity in top_peaks:
                ax.annotate(f'{mz:.2f}', xy=(mz, intensity), xytext=(0, 5),
                           textcoords='offset points', fontsize=8, ha='center',
                           rotation=90, va='bottom')
        
        plt.tight_layout()
        
        if output_file is not None:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f'Plot saved to: {output_file}')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_tic_with_peaks(self, output_file: Optional[str] = None, show: bool = True):
        """
        Plot the TIC with peaks, boundaries, and local maxima.
        
        Parameters
        ----------
        output_file : str, optional
            Output file path for the plot. If None, uses input filename with suffix.
        show : bool
            Whether to display the plot using plt.show()
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.tic_rt, self.tic_intensity, color='blue', linewidth=0.8, 
                alpha=0.7, label=f'{self.polarity_label} TIC')
        
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
        
        # annotate peak numbers
        for i, (rt, intensity) in enumerate(zip(apex_rt_list, apex_int_list)):
            ax.annotate(f'{i+1}', xy=(rt, intensity), xytext=(5, 10),
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        color='darkred', zorder=6)
        
        ax.set_xlabel('Retention Time (s)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        filename = os.path.basename(self.file_path)
        ax.set_title(f'{filename} - Peak Picking ({self.polarity_label} Only, Strict Boundaries)', fontsize=14, fontweight='bold')
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
    
    def __repr__(self) -> str:
        """String representation of the LCMSMeasMan."""
        return f"LCMSMeasMan(file_path='{self.file_path}', num_peaks={len(self.peaks)})"

