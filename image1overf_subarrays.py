#!/usr/bin/env python

"Chris Willott's 1/f removal script. https://github.com/chriswillott/jwst/blob/master/image1overf.py"

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.convolution import Gaussian2DKernel,convolve
from photutils import Background2D, MedianBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel
from jwst.datamodels import dqflags
from copy import deepcopy

def sub1fimaging(cal2hdulist,sigma_bgmask,sigma_1fmask,splitamps,usesegmask):
    """
    Determine a 1/f correction for JWST imaging data.
    Input is a level 2 calibrated NIRISS or NIRCam image.
    Subtracts a background and masks sources to determine 1/f stripes.
    Assumes stripes are constant along full detector row, unless splitamps set to True.
    Only does a correction for FULL (NIRCam and NIRISS) and SUB256 (NIRISS) subarrays.
    Uses 'SLOWAXIS' keyword to determine detector orientation.

    Output:
    Returns data array of corrected image, not including the reference pixels

    Parameters:
    cal2hdulist - hdulist of the input level 2 calibrated image
    sigma_bgmask - sigma of outliers for masking when making background  (suggested value=3.0)
    sigma_1fmask - sigma of outliers for masking when making 1/f correction  (suggested value=2.0)
    splitamps - fit each of the 4 amps separately when full-frame  (set this to True when the field is sparse)
    usesegmask - whether to use a segmentation image as the mask before fitting 1/f stripes (recommend set to True in most cases)
    """

    # (updates by DC 10/2/23)
    # Load data (and copy before correcting)
    data = cal2hdulist['SCI'].data + 0  # science data
    data = np.nan_to_num(data)  # replace nan with zeros for calculations
    dq   = cal2hdulist['DQ'].data  + 0  # data quality
    slowaxis = abs(cal2hdulist['PRIMARY'].header['SLOWAXIS'])

    # Extract data pixels without reference pixels
    reference_pixel_flag = dqflags.pixel['REFERENCE_PIXEL']

    reference_pixels = np.bitwise_and(dq, reference_pixel_flag)
    ny, nx = data.shape
    reference_rows = np.sum(reference_pixels[ny//2].astype(bool)) // 2
    
    data_columns = np.where(reference_pixels[ny//2], np.nan, np.arange(nx).astype(int))
    lo_column = int(np.nanmin(data_columns))
    hi_column = int(np.nanmax(data_columns))
    
    data_rows = np.where(reference_pixels[:,nx//2], np.nan, np.arange(ny).astype(int))
    lo_row = int(np.nanmin(data_rows))
    hi_row = int(np.nanmax(data_rows))
    
    data = data[lo_row:hi_row, lo_column:hi_column]
    dq   = dq[lo_row:hi_row, lo_column:hi_column]
    # (end updates)
    
    #Get a mask of bad pixels from the DQ array
    mask = np.zeros(data.shape,dtype=bool)
    i_yy,i_xx = np.where((np.bitwise_and(dq, dqflags.group['DO_NOT_USE']) == 1))
    mask[i_yy,i_xx] = 1

    #Make a flux distribution centered on the median that on the higher than median side is a reflection of the lower than median side.
    #Assumption here is not huge variation in background across field, so median is a useful value of typical background level
    #Note this mask is only used to generate the background model
    gooddata = data[np.where(mask==0)]
    median = np.median(gooddata)
    lowpixels = np.where(gooddata<median)
    bgpixels = np.concatenate(((gooddata[lowpixels]-median),(median-gooddata[lowpixels])))
    #Add to mask all pixels that have flux < or > 3sigma of this flux distribution 
    mask[np.where(data>(median+sigma_bgmask*np.std(bgpixels)))] = 1
    mask[np.where(data<(median-sigma_bgmask*np.std(bgpixels)))] = 1

    #testhdulist = deepcopy(cal2hdulist)
    #testdata =  deepcopy(data)
    #testdata[np.where(mask>0)] = 0.0
    #testhdulist['SCI'].data = testdata
    #testoutfile='testmaskinit.fits'
    #testhdulist.writeto(testoutfile,overwrite=True)    

    #Do a background subtraction to separate background variations from 1/f stripe determination
    sigma_clip_forbkg = SigmaClip(sigma=3., maxiters=5)
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(data, (34, 34),filter_size=(5, 5), mask=mask, sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator)
    except:
        bkg = Background2D(data, (34, 34),filter_size=(5, 5), mask=mask, sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator, exclude_percentile=30.)
    bksubdata = data-bkg.background

    #Remake mask on background-subtracted data, 
    #Two options - use segmentation region on smoothed image 
    #            - use same method as before but now clipping at < or > sigma_1fmask sigma of the flux distribution 
    mask = np.zeros(data.shape,dtype=bool)
    mask[i_yy,i_xx] = 1
    if usesegmask:
        # Before detection, smooth image with Gaussian FWHM = 5 pixels - this gives better balance between stars and BCGs because ghosts are more diffuse than stars
        kernel = make_2dgaussian_kernel(5.0, size=5)
        convolved_bksubdata = convolve(bksubdata, kernel)
        bksubdata_masked = np.ma.masked_array(bksubdata, mask=mask)
        clippedsigma = np.std(sigma_clip_forbkg(bksubdata_masked, masked=True, copy=False))
        #print (clippedsigma)
        photthreshold = sigma_1fmask*clippedsigma
        segm_detect = detect_sources(convolved_bksubdata, photthreshold, mask=mask, npixels=7)
        segimage = segm_detect.data.astype(np.uint32)
        mask[np.where(segimage>0)] = 1
    else:     
        gooddata = bksubdata[np.where(mask==0)]
        median = np.median(gooddata)
        negpixels = np.where(gooddata<median)
        bgpixels = np.concatenate(((gooddata[negpixels]-median),(median-gooddata[negpixels])))
        mask[np.where(bksubdata>(median+sigma_1fmask*np.std(bgpixels)))] = 1
        mask[np.where(bksubdata<(median-sigma_1fmask*np.std(bgpixels)))] = 1
    
    #testdata =  deepcopy(bksubdata)
    #testdata[np.where(mask>0)] = 0.0
    #testhdulist['SCI'].data = testdata
    #testhdulist['ERR'].data = segimage
    #testoutfile='testmaskpostbkg_seg_sigma_1fmask0p7.fits'
    #testhdulist.writeto(testoutfile,overwrite=True)    

    #Make masked array of background-subtracted data and then take median along columns (slowaxis=1) or rows (slowaxis=2)
    #if splitamps is True and full frame then define sections of 4 amps and fit and subtract each separately
    if splitamps==True and cal2hdulist['PRIMARY'].header['SUBARRAY']=='FULL':
        stripes=np.zeros(data.shape)
        #define sections - sub off 4 because reference pixels already removed
        loamp=np.array([4,512,1024,1536])-4
        hiamp=np.array([512,1024,1536,2044])-4
        numamps=loamp.size
        for k in range(numamps):
            if slowaxis==1:
                maskedbksubdata = np.ma.array(bksubdata[loamp[k]:hiamp[k],:], mask=mask[loamp[k]:hiamp[k],:])
                stripes[loamp[k]:hiamp[k],:] = np.ma.median(maskedbksubdata, axis=(slowaxis-1),keepdims=True)
            elif slowaxis==2:
                maskedbksubdata = np.ma.array(bksubdata[:,loamp[k]:hiamp[k]], mask=mask[:,loamp[k]:hiamp[k]])
                stripes[:,loamp[k]:hiamp[k]] = np.ma.median(maskedbksubdata, axis=(slowaxis-1),keepdims=True)
    else:        
        maskedbksubdata = np.ma.array(bksubdata, mask=mask)
        stripes = np.ma.median(maskedbksubdata, axis=(slowaxis-1),keepdims=True)

    correcteddata = data-stripes
    
    cal2hdulist['SCI'].data[lo_row:hi_row, lo_column:hi_column] = correcteddata

    return cal2hdulist['SCI']  # correcteddata
