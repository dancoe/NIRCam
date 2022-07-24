# NIRCam
Working with real JWST NIRCam images

https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline-overview/stages-of-jwst-data-processing

_uncal.fits (Level 1b) -- uncalibrated images

Stage 1: Detector1Pipeline: Count-rate slopes: Detector-level corrections and ramp fitting

_rate.fits (Level 2a) -- cosmic rays removed

Stage 2: Image2Pipeline: Calibrations: Instrument-mode: background subtraction and flat field correction

_cal.fits (Level 2b) -- galaxy wings appear

Stage 3: Combining multiple exposures within an observation

_i2d.fits (Level 2c) -- drizzled image
