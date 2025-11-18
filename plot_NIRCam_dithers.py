#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot NIRCam Dither Patterns and Depth Maps

This script visualizes JWST NIRCam dither patterns and creates depth maps showing
coverage across the field of view. It combines functionality for single pointings
and mosaic configurations.

Usage:
    # Old script style (positional + keyword arguments in any order)
    python plot_NIRCam_dithers.py fullbox 3tightgaps 5 mosaic long
    
    # Modern style (with --flags)
    python plot_NIRCam_dithers.py fullbox --pattern 3tightgaps --cmax 5 --mosaic --long
    
    # Mixed style
    python plot_NIRCam_dithers.py fullbox 3tightgaps --mosaic --long

Examples:
    python plot_NIRCam_dithers.py NircamImagingFull
    python plot_NIRCam_dithers.py fullbox 3TIGHT long
    python plot_NIRCam_dithers.py intramodule 6 --mosaic
    python plot_NIRCam_dithers.py fullbox 9 8 mosaic long path

Positional Arguments (old script style):
    dither_file              Dither file name (e.g., "full", "fullbox", "intrasca")
    pattern (optional)       Pattern to plot (e.g., "3tightgaps", "9")
    cmax (optional)          Color scale maximum (numeric)

Keywords (appear anywhere, any order):
    long                     Plot long wavelength channel
    mosaic                   Generate 3x3 mosaic
    path                     Show dither path points
    noshow                   Don't auto-open output files

Flag-Style Arguments (modern style):
    --pattern NAME           Specific pattern to plot
    --long                   Plot Long Wavelength Channel (default: Short)
    --mosaic                 Generate 3×3 mosaic pattern
    --mosaic2x1              Generate 2×1 mosaic pattern
    --mosaic2x1-offset       Generate 2×1 offset mosaic pattern
    --row-overlap N          Mosaic row overlap fraction (default: 0.0)
    --col-overlap N          Mosaic column overlap fraction (default: 0.0)
    --path                   Show dither path with points
    --max-dithers N          Maximum number of dithers to plot
    --cmax N                 Maximum value for color scale
    --xmax N                 Maximum X extent in arcmin (default: 4.5)
    --ymax N                 Maximum Y extent in arcmin (default: 2.75)
    --resolution FACTOR      Grid resolution multiplier (default: 1.0)
    --output-dir DIR         Output directory (default: plots/)
    --noshow                 Don't open output files automatically
"""

import sys
import os
import argparse
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.path
from matplotlib.patches import Polygon
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Plot Formatting Functions (extracted from coeplot3.py)
# ============================================================================

def setup_plot_style(fontsize=18, labsize=None, legsize=None, 
                     left=0.15, bottom=0.15, top=0.95, right=0.925, 
                     lw=2, axlw=1, ticksize=8, minorticksize=4):
    """Configure matplotlib plotting style with thick lines and nice formatting."""
    if labsize is None:
        labsize = fontsize
    if legsize is None:
        legsize = fontsize - 4
    
    params = {
        'axes.labelsize': labsize,
        'font.size': fontsize,
        'legend.fontsize': legsize,
        'figure.subplot.left': left,
        'figure.subplot.bottom': bottom,
        'figure.subplot.top': top,
        'figure.subplot.right': right,
        'lines.linewidth': lw,
        'axes.linewidth': axlw,
        'xtick.major.size': ticksize,
        'ytick.major.size': ticksize,
        'xtick.minor.size': minorticksize,
        'ytick.minor.size': minorticksize,
        'legend.numpoints': 1,
        'interactive': False,
    }
    plt.rcParams.update(params)


def multiples(lo, hi, d=None, n=4):
    """Generate nice tick mark values."""
    if d is None:
        d = (hi - lo) / n
        # Round to nice number
        mag = 10 ** np.floor(np.log10(d))
        d_norm = d / mag
        if d_norm <= 1:
            d = mag
        elif d_norm <= 2:
            d = 2 * mag
        elif d_norm <= 5:
            d = 5 * mag
        else:
            d = 10 * mag
    
    start = np.ceil(lo / d) * d
    end = np.floor(hi / d) * d
    return np.arange(start, end + d/2, d)


def savepng(filename, dpi=150):
    """Save figure as PNG."""
    if not filename.endswith('.png'):
        filename += '.png'
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")


def savefits(data, filename):
    """Save array as FITS file."""
    if not filename.endswith('.fits'):
        filename += '.fits'
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.writeto(filename, overwrite=True)
    print(f"Saved: {filename}")


# ============================================================================
# Dither Pattern Parsing
# ============================================================================

def parse_dither_file(filepath):
    """
    Parse NIRCam dither pattern file.
    
    Returns:
        dict: Dictionary mapping pattern names to arrays of (index, x, y) offsets
    """
    patterns = {}
    current_pattern = None
    current_offsets = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                if current_pattern and current_offsets:
                    # Save completed pattern
                    patterns[current_pattern] = np.array(current_offsets)
                    current_pattern = None
                    current_offsets = []
                continue
            
            parts = line.split()
            
            # Check if this is a pattern name (number or string)
            if len(parts) == 1:
                if current_pattern and current_offsets:
                    patterns[current_pattern] = np.array(current_offsets)
                    current_offsets = []
                current_pattern = parts[0]
            elif len(parts) == 3:
                # Parse offset line: index x y
                try:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    current_offsets.append([idx, x, y])
                except ValueError:
                    continue
        
        # Save final pattern
        if current_pattern and current_offsets:
            patterns[current_pattern] = np.array(current_offsets)
    
    return patterns


# ============================================================================
# NIRCam Field of View Loading
# ============================================================================

def load_nircam_fov(fov_dir='.'):
    """
    Load NIRCam field of view aperture definitions (hardcoded).
    
    Uses the first 10 apertures (NRCA1-5_FULL_OSS, NRCB1-5_FULL_OSS).
    
    Args:
        fov_dir: Unused, kept for API compatibility
    
    Returns:
        tuple: (allverts, allnames) where allverts is array of vertices in arcmin,
               allnames is list of aperture names
    """
    # First 10 apertures from NIRCam vertices.txt (in arcseconds)
    # Format: [X1, X2, X3, X4, X5, Y1, Y2, Y3, Y4, Y5]
    allverts_arcsec = np.array([
        [88.915, 153.138, 152.056, 88.712, 88.915, -559.864, -559.248, -495.167, -495.610, -559.864],
        [151.351, 88.700, 88.586, 151.933, 151.351, -427.976, -428.183, -491.469, -491.097, -427.976],
        [19.517, 84.035, 83.798, 20.171, 19.517, -560.196, -560.006, -495.756, -495.605, -560.196],
        [83.693, 20.823, 20.410, 83.980, 83.693, -428.216, -428.022, -491.569, -491.435, -428.216],
        [21.091, 151.256, 149.515, 22.429, 21.091, -558.387, -557.599, -428.838, -428.649, -558.387],
        [-89.541, -152.047, -152.803, -89.610, -89.541, -426.370, -426.028, -489.049, -489.556, -426.370],
        [-153.712, -89.573, -89.079, -152.348, -153.712, -557.111, -558.072, -493.879, -493.099, -557.111],
        [-21.954, -84.731, -84.544, -21.069, -21.954, -425.835, -426.415, -489.565, -489.301, -425.835],
        [-84.797, -20.346, -21.177, -84.742, -84.797, -557.970, -558.053, -493.487, -493.737, -557.970],
        [-25.740, -153.036, -154.551, -24.185, -25.740, -426.717, -426.959, -555.965, -556.655, -426.717],
    ])
    
    # Convert arcsec to arcmin
    allverts = allverts_arcsec / 60.0
    
    # Aperture names
    allnames = [
        'NRCA1_FULL_OSS',
        'NRCA2_FULL_OSS',
        'NRCA3_FULL_OSS',
        'NRCA4_FULL_OSS',
        'NRCA5_FULL_OSS',
        'NRCB1_FULL_OSS',
        'NRCB2_FULL_OSS',
        'NRCB3_FULL_OSS',
        'NRCB4_FULL_OSS',
        'NRCB5_FULL_OSS',
    ]
    
    return allverts, allnames


# ============================================================================
# Depth Map Calculation
# ============================================================================

class DepthMapCalculator:
    """Calculate exposure depth maps for NIRCam dither patterns."""
    
    def __init__(self, xhi: float = 4.5, yhi: float = 2.75, resolution: float = 1, fov_dir: str = '.'):
        """
        Initialize depth map calculator.
        
        Args:
            xhi: Maximum x extent in arcmin (xlo = -xhi)
            yhi: Maximum y extent in arcmin (ylo = -yhi)
            resolution: Grid resolution multiplier (higher = finer grid, slower)
            fov_dir: Directory containing FOV files
        """
        self.xhi = xhi
        self.yhi = yhi
        self.xlo = -xhi
        self.ylo = -yhi
        
        # Create grid - use reasonable resolution (1 = ~3000x1500 pixels)
        nx = int((self.xhi - self.xlo) * 60 * 5 * resolution)
        ny = int((self.yhi - self.ylo) * 60 * 5 * resolution)
        
        print(f"Grid size: {nx} x {ny}")
        
        self.yyy, self.xxx = np.mgrid[self.ylo:self.yhi:ny*1j, 
                                       self.xlo:self.xhi:nx*1j]
        self.imagecoords = np.array([self.xxx.ravel(), self.yyy.ravel()]).T
        
        # Load FOV data from files
        self.allverts, self.allnames = load_nircam_fov(fov_dir)
        
        # Reference position (V2, V3 in arcsec, convert to arcmin)
        self.xref = -0.32 / 60.0
        self.yref = -492.59 / 60.0
        
        # Mosaic tile size (arcmin)
        # From JWST docs: tile extent is 5.115033' x 2.221150'
        self.mosaicx = 5.115033
        self.mosaicy = 2.221150
    
    def calculate_depth(self, offsets, plot_long=False, mosaic_config=None):
        """
        Calculate exposure depth map for given dither offsets.
        
        Args:
            offsets: Nx3 array of (index, x_arcsec, y_arcsec)
            plot_long: If True, plot long wavelength channel; else short
            mosaic_config: Tuple of (row_overlap, col_overlap, mosaic_type)
                          mosaic_type in ['3x3', '2x1', '2x1-offset']
        
        Returns:
            2D array of exposure depths
        """
        obs = np.zeros_like(self.xxx, dtype=np.int16)
        
        # Extract x, y offsets (convert arcsec to arcmin)
        xx = offsets[:, 1] / 60.0
        yy = offsets[:, 2] / 60.0
        
        # Mosaic configuration
        if mosaic_config:
            row_overlap, col_overlap, mosaic_type = mosaic_config
            mx = self.mosaicx * (1 - col_overlap)
            my = self.mosaicy * (1 - row_overlap)
            
            if mosaic_type == '3x3':
                tile_positions = [(iy, ix) for iy in (-1, 0, 1) for ix in (-1, 0, 1)]
            elif mosaic_type == '2x1':
                tile_positions = [(0.5, 0), (-0.5, 0)]
            elif mosaic_type == '2x1-offset':
                tile_positions = [(0.5, -0.5), (-0.5, 0.5)]
            else:
                tile_positions = [(0, 0)]
        else:
            tile_positions = [(0, 0)]
            mx = my = 0
        
        # Loop over mosaic tiles
        for iy, ix in tile_positions:
            # Loop over dither positions
            for i in range(len(xx)):
                # Add tile offset to dither position
                dx = xx[i] + ix * mx
                dy = yy[i] + iy * my
                
                # Add contributions from each aperture
                obs = self._add_aperture_contributions(obs, dx, dy, plot_long)
        
        return obs
    
    def _add_aperture_contributions(self, obs, dx, dy, plot_long):
        """Add aperture contributions to observation map (matches old script logic)."""
        # Loop through first 10 apertures (NRCA1-5_FULL_OSS, NRCB1-5_FULL_OSS)
        # Indices 10-12 contain duplicates/special apertures (NRCALL, NRCAS, NRCA1_FULL)
        for i in range(min(10, len(self.allnames))):
            name = self.allnames[i]
            
            # Skip subarrays and special apertures  
            if 'SUB' in name:
                continue
            if 'S_' in name:
                continue
            if 'ALL_' in name:
                continue
            
            # Get vertices for this aperture
            verts = self.allverts[i].copy()
            # Data format: [X1,X2,X3,X4,X5, Y1,Y2,Y3,Y4,Y5]
            # Take first 4 X coords and first 4 Y coords (skip closing vertex)
            xv = verts[:4]
            yv = verts[5:9]
            
            # Apply reference position offset
            xv = xv - self.xref
            yv = yv - self.yref
            
            # Flip V3 onto x-axis (matching old script)
            xv = -xv
            
            # Apply dither offset
            xv = xv + dx
            yv = yv + dy
            
            # Reconstruct vertices
            verts = np.array([xv, yv])
            
            # Check wavelength channel
            if name[4] == '5':  # Long Wavelength (NRCA5, NRCB5)
                if not plot_long:
                    continue
            else:  # Short Wavelength
                if plot_long:
                    continue
            
            # Check which pixels fall inside this aperture
            path = matplotlib.path.Path(verts.T)
            inside = path.contains_points(self.imagecoords)
            inside = inside.reshape(self.xxx.shape).astype(np.int16)
            
            obs += inside
        
        return obs


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_depth_map(xxx, yyy, obs, title, max_depth=None, plot_long=False, 
                   xlo=-4.5, xhi=4.5, ylo=-2.75, yhi=2.75):
    """
    Plot exposure depth map (matching old script exactly).
    
    Args:
        xxx, yyy: 2D coordinate grids (arcmin)
        obs: 2D exposure depth array
        title: Plot title
        max_depth: Maximum depth for color scale (default: max of data)
        plot_long: If True, plotting long wavelength channel
        xlo, xhi, ylo, yhi: Axis limits (arcmin)
    """
    # Set up colormap
    cmap = 'CMRmap_r'
    
    plt.imshow(obs, extent=[xlo, xhi, ylo, yhi],
               origin='lower', interpolation='nearest', cmap=cmap,
               aspect='auto', vmin=0, vmax=max_depth or obs.max())
    
    # Colorbar - match old script exactly
    ndithers = max_depth or int(obs.max())
    if ndithers < 26:
        cticks = multiples(0, ndithers, d=1)
    else:
        cticks = multiples(0, ndithers, d=3)
    cticks = cticks[cticks <= ndithers]
    
    cbar = plt.colorbar(ticks=cticks, shrink=1, pad=0.035, fraction=0.075)
    
    # Labels
    plt.xlabel('X Ideal (arcmin)')
    plt.ylabel('Y Ideal (arcmin)')
    
    channel = 'Long' if plot_long else 'Short'
    cbar.set_label('Exposures ({0} Wavelength Channel)'.format(channel), 
                   rotation=90, labelpad=20, fontsize=16)
    
    # Set aspect ratio first
    plt.gca().set_aspect('equal', 'datalim')
    
    # Set limits and ticks to match old script
    # Ticks at 1 arcmin intervals
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    
    xticks_vals = np.arange(np.ceil(xlo), np.floor(xhi) + 0.5, 1)
    yticks_vals = np.arange(np.ceil(ylo), np.floor(yhi) + 0.5, 1)
    plt.xticks(xticks_vals)
    plt.yticks(yticks_vals)
    
    # Move ticks inside the frame
    ax = plt.gca()
    ax.tick_params(direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    plt.title(title)


def plot_dither_path(offsets, max_dithers=None):
    """
    Overlay dither path on existing plot.
    
    Args:
        offsets: Nx3 array of (index, x_arcsec, y_arcsec)
        max_dithers: Maximum number of dithers to show
    """
    xx = offsets[:, 1] / 60.0  # Convert to arcmin
    yy = offsets[:, 2] / 60.0
    
    n = max_dithers or len(xx)
    
    for i in range(n):
        if i == 0:
            label = 'Dither Path'
        else:
            label = None
        plt.plot(xx[i], yy[i], 'go', mec='g', ms=6, zorder=10, 
                alpha=0.7, label=label)
    
    plt.legend(loc='best', framealpha=0.9)


# ============================================================================
# Main Function
# ============================================================================

def parse_args_intelligently(argv):
    """
    Parse command line arguments intelligently, supporting both positional and keyword arguments.
    
    Mimics the original script behavior:
    - First arg: dither_file (required)
    - Second arg: pattern (optional, e.g., "3tightgaps", "9")
    - Third arg: cmax (optional, numeric)
    - Keywords anywhere: "long", "mosaic", "path", "noshow", "show", "nonum", etc.
    
    Args:
        argv: sys.argv[1:] (excluding program name)
    
    Returns:
        Namespace object with parsed arguments
    """
    import argparse
    
    # Create a minimal parser for help and basic validation
    parser = argparse.ArgumentParser(
        description='Plot JWST NIRCam dither patterns and depth maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)  # We'll handle help manually to support our custom parsing
    
    # Parse known args first to extract positional arguments
    # This is a simple approach: extract positional args in order
    
    class Args:
        def __init__(self):
            self.dither_file: str = ''
            self.pattern: str = ''
            self.long: bool = False
            self.mosaic: bool = False
            self.mosaic2x1: bool = False
            self.mosaic2x1_offset: bool = False
            self.row_overlap: float = 0.0
            self.col_overlap: float = 0.0
            self.path: bool = False
            self.max_dithers: Optional[int] = None
            self.cmax: Optional[int] = None
            self.resolution: float = 1.0
            self.xmax: float = 4.5
            self.ymax: float = 2.75
            self.noshow: bool = False
            self.output_dir: str = '.'
            self.show: bool = False
    
    args = Args()
    
    # First pass: check for keywords anywhere in argv (like old script)
    argv_str = ' '.join(argv).lower()
    if 'long' in argv_str:
        args.long = True
    if 'mosaic' in argv_str:
        args.mosaic = True
    if 'path' in argv_str:
        args.path = True
    if 'noshow' in argv_str:
        args.noshow = True
    if 'show' in argv_str:
        args.show = True
    
    # Second pass: extract positional arguments and handle explicit flags
    positional_idx = 0
    i = 0
    
    while i < len(argv):
        arg = argv[i]
        
        # Check for help
        if arg in ['-h', '--help']:
            parser.print_help()
            sys.exit(0)
        
        # Check for named arguments with values
        if arg in ['--output-dir', '--row-overlap', '--col-overlap', '--resolution', '--max-dithers', '--cmax', '--xmax', '--ymax']:
            if i + 1 < len(argv):
                val = argv[i + 1]
                if arg == '--output-dir':
                    args.output_dir = val
                elif arg == '--row-overlap':
                    args.row_overlap = float(val)
                elif arg == '--col-overlap':
                    args.col_overlap = float(val)
                elif arg == '--resolution':
                    args.resolution = float(val)
                elif arg == '--max-dithers':
                    args.max_dithers = int(val)
                elif arg == '--cmax':
                    args.cmax = int(val)
                elif arg == '--xmax':
                    args.xmax = float(val)
                elif arg == '--ymax':
                    args.ymax = float(val)
                i += 2
                continue
        
        # Check for boolean flags (--flag style)
        if arg in ['--long', '--mosaic', '--mosaic2x1', '--mosaic2x1-offset', '--path', '--noshow', '--show']:
            if arg == '--long':
                args.long = True
            elif arg == '--mosaic':
                args.mosaic = True
            elif arg == '--mosaic2x1':
                args.mosaic2x1 = True
            elif arg == '--mosaic2x1-offset':
                args.mosaic2x1_offset = True
            elif arg == '--path':
                args.path = True
            elif arg == '--noshow':
                args.noshow = True
            elif arg == '--show':
                args.show = True
            i += 1
            continue
        
        # Check for single-letter or short keyword flags (matching old script)
        # These are handled in first pass, only check flags here
        # If not a flag, treat as positional
        if not arg.startswith('-'):
            if positional_idx == 0:
                args.dither_file = arg
                positional_idx += 1
            elif positional_idx == 1:
                # Second arg: Try pattern first, fall back to max_dithers if all digits
                # This will be refined later when we know what patterns are available
                args.pattern = arg
                # Note: if it's all digits, this will be treated as max_dithers in second pass
                positional_idx += 1
            elif positional_idx == 2:
                # Third arg: should be cmax (numeric)
                if arg.isdigit():
                    args.cmax = int(arg)
                positional_idx += 1
        
        i += 1
    
    # Validate
    if not args.dither_file:
        print("Error: dither_file argument is required")
        print("\nUsage: python plot_NIRCam_dithers.py <dither_file> [pattern] [cmax] [options]")
        print("\nExamples:")
        print("  python plot_NIRCam_dithers.py fullbox")
        print("  python plot_NIRCam_dithers.py fullbox 3tightgaps")
        print("  python plot_NIRCam_dithers.py fullbox 3tightgaps 5 mosaic long")
        print("  python plot_NIRCam_dithers.py fullbox 8 --mosaic --long")
        sys.exit(1)
    
    return args


def main():
    """Main execution function."""
    # Parse arguments intelligently (supporting both old style and new style)
    args = parse_args_intelligently(sys.argv[1:])
    
    # Setup plotting style
    setup_plot_style(bottom=0.135, top=0.9, left=0.1, right=0.96)
    
    # Find dither file - support short names like "full", "fullbox", "intrasca"
    dither_file = args.dither_file
    
    # Check in NIRCam_dithers directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dither_dir = os.path.join(os.path.dirname(script_dir), 'NIRCam_dithers')
    
    # Try exact match first
    if not dither_file.endswith('.txt'):
        dither_file_txt = dither_file + '.txt'
    else:
        dither_file_txt = dither_file
    
    dither_path = os.path.join(dither_dir, dither_file_txt)
    
    # If not found, try pattern matching for short names
    if not os.path.exists(dither_path):
        # Try common pattern: NircamImaging + capitalized name
        search_name = dither_file.replace('.txt', '')
        candidates = [
            f'NircamImaging{search_name.title()}.txt',
            f'NircamImaging{search_name.upper()}.txt',
            f'NircamImaging{search_name.capitalize()}.txt',
            f'Nircam{search_name.title()}.txt',
            f'Nircam{search_name.upper()}.txt',
        ]
        
        for candidate in candidates:
            test_path = os.path.join(dither_dir, candidate)
            if os.path.exists(test_path):
                dither_path = test_path
                dither_file_txt = candidate
                print(f"Found: {candidate}")
                break
    
    if not os.path.exists(dither_path):
        print(f"Error: Dither file not found: {dither_path}")
        print(f"\nTried: {dither_file}")
        print(f"\nAvailable files in {dither_dir}:")
        try:
            for f in sorted(os.listdir(dither_dir)):
                if f.endswith('.txt'):
                    print(f"  {f}")
        except:
            pass
        sys.exit(1)
    
    # Parse dither patterns
    print(f"Reading dither patterns from: {dither_path}")
    patterns = parse_dither_file(dither_path)
    print(f"Found {len(patterns)} patterns: {list(patterns.keys())}")
    
    # Handle case where second arg was all digits (treat as max_dithers, not pattern)
    if args.pattern and args.pattern.isdigit():
        # It's all digits - treat as max_dithers if no pattern was found
        args.max_dithers = int(args.pattern)
        args.pattern = ''
    
    # Filter patterns if requested
    if args.pattern:
        # Try exact match first (case-insensitive)
        pattern_key = None
        for key in patterns.keys():
            if key.upper() == args.pattern.upper():
                pattern_key = key
                break
        
        if pattern_key:
            patterns = {pattern_key: patterns[pattern_key]}
        else:
            print(f"Error: Pattern '{args.pattern}' not found")
            print(f"Available patterns: {list(patterns.keys())}")
            sys.exit(1)
    
    # Initialize depth calculator (resolution=1 for speed, increase for finer detail)
    # FOV files should be in the scripts directory
    calculator = DepthMapCalculator(xhi=args.xmax, yhi=args.ymax, 
                                   resolution=args.resolution,
                                   fov_dir=script_dir)
    
    # Determine mosaic configuration
    mosaic_config = None
    if args.mosaic:
        mosaic_config = (args.row_overlap, args.col_overlap, '3x3')
    elif args.mosaic2x1_offset:
        mosaic_config = (args.row_overlap, args.col_overlap, '2x1-offset')
    elif args.mosaic2x1:
        mosaic_config = (args.row_overlap, args.col_overlap, '2x1')
    
    # Set default output directory to plots/ subdirectory
    if args.output_dir == '.':
        args.output_dir = 'plots'
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each pattern
    # Use short name for output (like old script: fullbox8nirspec not NircamImagingFullBox_8NIRSPEC)
    base_filename_full = os.path.splitext(os.path.basename(dither_file_txt))[0]
    # Extract short name: NircamImagingFullBox -> fullbox
    if base_filename_full.startswith('NircamImaging'):
        base_filename = base_filename_full.replace('NircamImaging', '').lower()
    elif base_filename_full.startswith('Nircam'):
        base_filename = base_filename_full.replace('Nircam', '').lower()
    else:
        base_filename = args.dither_file.lower().replace('.txt', '')
    
    for pattern_name in sorted(patterns.keys()):
        offsets = patterns[pattern_name]
        
        # Limit number of dithers if requested
        if args.max_dithers:
            offsets = offsets[:args.max_dithers]
        
        ndithers = len(offsets)
        print(f"\nProcessing pattern: {pattern_name} ({ndithers} dithers)")
        
        # Calculate depth map
        obs = calculator.calculate_depth(offsets, 
                                         plot_long=args.long,
                                         mosaic_config=mosaic_config)
        
        # Create figure with specific margins (matching old script)
        # Old script: 1000 x 600 pixels at 100 DPI = 10 x 6 inches
        fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.subplots_adjust(bottom=0.135, top=0.9, left=0.1, right=0.96)
        
        # Build title - include dither count if max_dithers was specified
        #title = f'NIRCam Primary Dithers: {base_filename.upper()} {pattern_name}'
        title = f'NIRCam Primary Dithers: {pattern_name}'
        if args.max_dithers:
            title += f' {ndithers}'
        if mosaic_config:
            title = title.replace('NIRCam', 'NIRCam Mosaic')
        
        # Plot depth map
        plot_depth_map(calculator.xxx, calculator.yyy, obs, title,
                      max_depth=args.cmax, plot_long=args.long,
                      xlo=calculator.xlo, xhi=calculator.xhi,
                      ylo=calculator.ylo, yhi=calculator.yhi)
        
        # Optionally overlay dither path
        if args.path:
            plot_dither_path(offsets, max_dithers=args.max_dithers)
        
        # Build output filename - append dither count if max_dithers was specified
        outname = f"{base_filename}{pattern_name.lower()}"
        if args.max_dithers:
            outname += str(args.max_dithers)
        if args.long:
            outname += '_long'
        if mosaic_config:
            mtype = mosaic_config[2].replace('-', '_')
            outname += f'_mosaic_{mtype}'
            if args.row_overlap != 0 or args.col_overlap != 0:
                outname += f'_overlap_{args.row_overlap}_{args.col_overlap}'
        
        outpath = os.path.join(args.output_dir, outname)
        
        # Save outputs
        savepng(outpath)
        savefits(obs, outpath)
        
        # Show plot if requested
        if not args.noshow:
            if sys.platform == 'darwin':  # macOS
                os.system(f'open {outpath}.png')
        
        plt.close(fig)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
