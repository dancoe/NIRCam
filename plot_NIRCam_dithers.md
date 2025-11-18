# NIRCam Dither Pattern Visualization Tool

Visualize JWST NIRCam dither patterns and generates exposure depth maps showing coverage across the field of view.

![](fullbox6tight.png)

## Features

- **Dither Pattern Visualization**: Display coverage maps for any NIRCam dither pattern
- **Exposure Depth Maps**: Shows how many exposures hit each pixel in the focal plane
- **Wavelength Channel Selection**: Plot short or long wavelength detector coverage separately
- **Mosaic Support**: Generate 3×3, 2×1, and offset mosaic patterns with configurable overlap
- **Flexible Pattern Selection**: Specify single patterns or process all patterns in a file
- **Publication-Quality Output**: PNG and FITS file outputs with exact formatting match to original script

## Requirements

###  Dither Patterns Files

Download and unzip the dither pattern files
via [JDox NIRCam Primary Dithers](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-operations/nircam-dithers-and-mosaics/nircam-primary-dithers):

* https://jwst-docs.stsci.edu/files/216457358/216457359/1/1762453956617/NIRCamDitherPatterns.zip

### Python Libraries

- Python 3.6+
- numpy
- matplotlib
- astropy

```bash
pip install numpy matplotlib astropy
```

## Usage

### Basic Usage

Plot all patterns from a dither file:
```bash
python plot_NIRCam_dithers.py fullbox
```

Plot a specific pattern:
```bash
python plot_NIRCam_dithers.py fullbox 3tightgaps
```

Limit to first 4 dithers only:
```bash
python plot_NIRCam_dithers.py intramodulebox 4
```

Plot with options:
```bash
python plot_NIRCam_dithers.py fullbox 3tightgaps 5 mosaic long
```

Plot with modern flag style:
```bash
python plot_NIRCam_dithers.py fullbox 3tightgaps --cmax 5 --mosaic --long
```

**Output Files**: By default, all plots are saved to the `plots/` subdirectory. Use `--output-dir` to change this.

### Argument Styles

The script supports both positional arguments (like the old script) and modern flag-style arguments:

**Old Script Style** (positional + keywords):
```bash
python plot_NIRCam_dithers.py fullbox 3tightgaps 5 mosaic long
```
- Position 1: `dither_file` (required)
- Position 2: `pattern` (optional, e.g., "3tightgaps", "9")
- Position 3: `cmax` (optional, color scale max)
- Keywords: `long`, `mosaic`, `path`, `noshow`, etc. (can appear anywhere)

**Modern Style** (with --flags):
```bash
python plot_NIRCam_dithers.py fullbox --pattern 3tightgaps --cmax 5 --mosaic --long
```

**Mixed Style** (positional + flags):
```bash
python plot_NIRCam_dithers.py fullbox 3tightgaps --cmax 5 --mosaic --long
```

### Command Line Options

```
Positional Arguments:
  dither_file              Dither file name (e.g., "full", "fullbox", "intrasca")
  pattern (optional)       Specific pattern to plot (e.g., "9", "3TIGHT", "3tightgaps")
  cmax (optional)          Maximum value for color scale (numeric, e.g., "5")

Optional Arguments (flag style):
  --pattern NAME          Specific pattern to plot (modern flag style)
  --long                  Plot Long Wavelength Channel (default: Short)
  --mosaic                Generate 3×3 mosaic pattern
  --mosaic2x1             Generate 2×1 mosaic pattern
  --mosaic2x1-offset      Generate 2×1 offset mosaic pattern
  --row-overlap N         Mosaic row overlap fraction (default: 0.0)
  --col-overlap N         Mosaic column overlap fraction (default: 0.0)
  --path                  Show dither path with points
  --max-dithers N         Maximum number of dithers to plot
  --cmax N                Maximum value for color scale
  --resolution FACTOR     Grid resolution multiplier (default: 1.0, higher=finer/slower)
  --output-dir DIR        Output directory (default: plots/)
  --noshow                Don't open output files automatically

Keywords (appear anywhere, any order):
  long                    Plot long wavelength channel
  mosaic                  Generate 3x3 mosaic
  path                    Show dither path
  noshow                  Don't auto-open output files
```

### Examples

**Old script style commands:**
```bash
# Plot "2TIGHTGAPS" pattern from the FULLBOX dither file
python plot_NIRCam_dithers.py fullbox 2tightgaps

# Plot long wavelength channel
python plot_NIRCam_dithers.py fullbox 3tightgaps long

# Generate a 3×3 mosaic
python plot_NIRCam_dithers.py fullbox 6 mosaic

# Limit to first 4 dithers (all patterns)
python plot_NIRCam_dithers.py intramodulebox 4

# Plot first 8 dithers with custom color scale
python plot_NIRCam_dithers.py fullbox 8 5

# All options combined (any order)
python plot_NIRCam_dithers.py fullbox 3tightgaps 5 mosaic long path noshow
```

**Modern flag style:**
```bash
# Plot "2TIGHTGAPS" pattern from the FULLBOX dither file
python plot_NIRCam_dithers.py fullbox --pattern 2tightgaps

# Generate a 3×3 mosaic with 10% row overlap and 20% column overlap
python plot_NIRCam_dithers.py fullbox 6 --mosaic --row-overlap 0.1 --col-overlap 0.2

# Plot with dither path overlay
python plot_NIRCam_dithers.py fullbox 2tightgaps --path

# Limit to first 4 dithers with custom color scale
python plot_NIRCam_dithers.py fullbox 8 --max-dithers 4 --cmax 4

# High resolution grid (slower but more detailed)
python plot_NIRCam_dithers.py fullbox 3tightgaps --resolution 2

# Save to custom directory
python plot_NIRCam_dithers.py fullbox 2tightgaps --output-dir ./my_plots
```

## Output Files

For each pattern, the script generates:
- **PNG image**: Publication-quality depth map visualization
- **FITS file**: Raw exposure depth data (1000×600 pixels at 100 DPI)

Output filenames follow the pattern: `{dither_type}{pattern_name}[_long][_mosaic_type][_overlap_params]`

Examples:
- `fullbox2tightgaps.png` - Short wavelength FULLBOX 2TIGHTGAPS pattern
- `fullbox2tightgaps_long.fits` - Long wavelength FITS data
- `intramodule6_mosaic_3x3_overlap_0.1_0.2.png` - 3×3 mosaic with specified overlap

## Output Appearance

The visualization matches the original JWST NIRCam dither visualization script exactly:
- **Plot size**: 1000×600 pixels
- **Colormap**: CMRmap_r (reversed rainbow colormap)
- **Axis ticks**: 1 arcmin intervals, positioned inside the frame
- **Margin settings**: Optimized for publication-quality figures
- **Color scale**: Automatic (number of dithers) with manual override via `--cmax`

## Supported Dither Files

The script supports dither pattern files in the `NIRCam_dithers/` directory:
- `NircamImagingFull.txt` (short: "full")
- `NircamImagingFullBox.txt` (short: "fullbox")
- `NircamImagingCompromiseSubpixel.txt` (short: "compromisesubpixel")
- `NircamImagingIntramodule.txt` (short: "intramodule")
- `NircamImagingIntrasca.txt` (short: "intrasca")
- `NircamImagingSubpixel.txt` (short: "subpixel")
- `NircamWfscPhasing.txt` (short: "wfscphasing")
- `NircamWfssSubpixel.txt` (short: "wfsssubpixel")
- And others

## Technical Details

### Coordinate System
- **Coordinate frame**: NIRCam Ideal (approximately aligned with JWST V2/V3)
- **Units**: Arcminutes
- **Reference position**: V2 = -0.32", V3 = -492.59" (converted to arcmin)
- **Axis orientation**: X Ideal (≈ -V2), Y Ideal (≈ V3)

### Detector Coverage
- **Primary apertures**: 10 NIRCam modules (NRCA1-5, NRCB1-5)
- **Wavelength channels**: 
  - Short (NRCA1-4, NRCB1-4)
  - Long (NRCA5, NRCB5)
- **Aperture vertices**: Hardcoded from actual JWST aperture definitions

### Performance
- **Default grid resolution**: 1× (≈3600×1800 pixels, ~5 seconds per pattern)
- **Resolution scaling**: Use `--resolution 2` for 2× finer detail (slower), or `0.5` for coarser/faster
- **Output**: PNG rendering at 100 DPI for consistent sizing

## Hardcoded Data

The script includes hardcoded vertex coordinates for the 10 primary NIRCam apertures. These are extracted from the official JWST aperture definitions and stored directly in the script, eliminating the need for external data files.

## Comparison with Original Script

This script modernizes the original `plotditherdepthfull.py` and `plotditherdepthfullmosaic2.py` by:
- **Python 3**: Modern Python syntax and libraries
- **Combined functionality**: Single script handles both single-pointing and mosaic patterns
- **Improved CLI**: Intuitive positional arguments and option names
- **Self-contained**: No external FOV data files needed (hardcoded apertures)
- **Exact compatibility**: Output appearance and numbering matches original scripts exactly

## Troubleshooting

**Pattern not found**: Check the dither file name and available patterns
```bash
python plot_NIRCam_dithers.py fullbox  # List all patterns
```

**Output files not opening**: Use `--noshow` to suppress automatic opening, or open manually:
```bash
open fullbox2tightgaps.png
```

**Slow execution**: Reduce resolution or limit number of dithers
```bash
python plot_NIRCam_dithers.py fullbox 2tightgaps --resolution 0.5
```

**Memory issues with high resolution**: Reduce grid resolution or output DPI
```bash
python plot_NIRCam_dithers.py fullbox 2tightgaps --resolution 0.5
```

## References

- JWST Dither and Mosaic Documentation: 
  * https://jwst-docs.stsci.edu/near-infrared-camera/nircam-operations/nircam-dithers-and-mosaics/
- NIRCam Aperture Definitions: JWST SIAF aperture reference files

