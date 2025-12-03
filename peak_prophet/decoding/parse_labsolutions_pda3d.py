"""
Parser for LabSolutions PDA 3D export files.

This script extracts the [PDA 3D] section from LabSolutions export files
and converts it to a format compatible with mocca2's Chromatogram class.
"""

import numpy as np
import re
from typing import Tuple


def parse_labsolutions_pda3d(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse LabSolutions PDA 3D section from export file.
    
    Parameters
    ----------
    file_path : str
        Path to the LabSolutions export file
        
    Returns
    -------
    data2d : np.ndarray
        2D array of shape (n_time, n_wavelength) containing intensity values
    time : np.ndarray
        1D array of time values in minutes
    wavelength : np.ndarray
        1D array of wavelength values in nm
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find the [PDA 3D] section
    pda3d_start = None
    for i, line in enumerate(lines):
        if '[PDA 3D]' in line:
            pda3d_start = i
            break
    
    if pda3d_start is None:
        raise ValueError("Could not find [PDA 3D] section in file")
    
    # Parse metadata
    metadata = {}
    header_line_idx = None
    data_start_line = None
    for i in range(pda3d_start + 1, min(pda3d_start + 20, len(lines))):
        line = lines[i].strip()
        if line.startswith('R.Time (min)'):
            header_line_idx = i
            data_start_line = i + 1  # Data starts on next line
            break
        if '\t' in line:
            key, value = line.split('\t', 1)
            try:
                # Try to convert to float
                metadata[key] = float(value)
            except ValueError:
                metadata[key] = value
    
    if data_start_line is None or header_line_idx is None:
        raise ValueError("Could not find data start line in [PDA 3D] section")
    
    # Extract wavelength values from header row (line after "R.Time (min)")
    header_line = lines[data_start_line].rstrip()
    header_parts = header_line.split('\t')
    
    # Wavelength values appear to be encoded as integers (wavelength * 100)
    # Skip empty parts at the beginning
    wavelength_encoded = []
    for part in header_parts:
        part = part.strip()
        if part and part.isdigit():
            wavelength_encoded.append(int(part))
    
    if not wavelength_encoded:
        raise ValueError(f"Could not extract wavelength values from header row. Parts: {header_parts[:10]}")
    
    wavelength = np.array(wavelength_encoded) / 100.0  # Convert to nm
    
    # Parse data rows
    time_values = []
    intensity_data = []
    
    for i in range(data_start_line + 1, len(lines)):
        line = lines[i].strip()
        
        # Stop if we hit another section (starts with '[')
        if line.startswith('['):
            break
        
        # Skip empty lines
        if not line:
            continue
        
        # Parse row: first value is time, rest are intensities
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        try:
            time_val = float(parts[0].strip())
            # Get intensities - need to match the number of wavelengths
            intensities = []
            for part in parts[1:len(wavelength_encoded)+1]:
                part = part.strip()
                if part:
                    intensities.append(float(part))
                else:
                    intensities.append(0.0)  # Default to 0 if empty
            
            # Make sure we have the right number of intensities
            if len(intensities) == len(wavelength_encoded):
                time_values.append(time_val)
                intensity_data.append(intensities)
        except (ValueError, IndexError) as e:
            # Skip malformed rows
            continue
    
    if not time_values:
        raise ValueError("No valid data rows found in [PDA 3D] section")
    
    # Convert to numpy arrays
    time = np.array(time_values)
    data2d = np.array(intensity_data)
    
    # Verify dimensions
    if data2d.shape != (len(time), len(wavelength)):
        raise ValueError(
            f"Data shape mismatch: expected ({len(time)}, {len(wavelength)}), "
            f"got {data2d.shape}"
        )
    
    print(f"Parsed PDA 3D data:")
    print(f"  Time points: {len(time)}")
    print(f"  Wavelength points: {len(wavelength)}")
    print(f"  Time range: {time[0]:.3f} - {time[-1]:.3f} min")
    print(f"  Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} nm")
    print(f"  Data shape: {data2d.shape}")
    
    return data2d, time, wavelength


def convert_to_mocca2_format(data2d: np.ndarray, time: np.ndarray, wavelength: np.ndarray) -> dict:
    """
    Convert parsed data to format expected by mocca2.
    
    mocca2 expects a dictionary with 'time', 'wavelength', and 'data' keys.
    """
    return {
        'time': time,
        'wavelength': wavelength,
        'data': data2d
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_labsolutions_pda3d.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        data2d, time, wavelength = parse_labsolutions_pda3d(input_file)
        
        # If output file specified, save as CSV
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
            # Save in a simple CSV format that mocca2 can read
            # Format: first row is wavelength header, first column is time
            with open(output_file, 'w') as f:
                # Write header: empty first cell, then wavelengths
                f.write('Time (min)')
                for wl in wavelength:
                    f.write(f',{wl:.1f}')
                f.write('\n')
                
                # Write data rows
                for i, t in enumerate(time):
                    f.write(f'{t:.6f}')
                    for intensity in data2d[i, :]:
                        f.write(f',{intensity}')
                    f.write('\n')
            
            print(f"\nSaved CSV file: {output_file}")
        else:
            print("\nParsing successful! Use this data with mocca2:")
            print("  from parse_labsolutions_pda3d import parse_labsolutions_pda3d")
            print("  from mocca2 import Chromatogram")
            print("  data2d, time, wavelength = parse_labsolutions_pda3d('uvdata30.txt')")
            print("  chrom = Chromatogram(sample={'time': time, 'wavelength': wavelength, 'data': data2d})")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

