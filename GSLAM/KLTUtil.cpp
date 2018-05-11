//
//  KLT.cpp
//  GSLAM
//
//  Created by ctang on 9/8/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "KLTUtil.h"
#include "error.h"

static const int _mindist = 10;
static const int _window_size = 21;
static const int _min_eigenvalue = 200;
static const float _min_determinant = 0.01f;
static const float _min_displacement = 0.1f;
static const int   _max_iterations = 10;
static const float _max_residue = 10.0f;


static const float _grad_sigma = 1.0f;
static const float _smooth_sigma_fact = 0.1f;
static const float _pyramid_sigma_fact = 0.9f;


//static const float _grad_sigma = 0.5f;
//static const float _smooth_sigma_fact = 0.1f;
//static const float _pyramid_sigma_fact = 0.3f;


static const float _step_factor = 1.0f;
static const bool  _sequentialMode = false;
static const bool  _lighting_insensitive = false;

/* for affine mapping*/
static const int _affineConsistencyCheck = -1;
static const int _affine_window_size = 15;
static const int _affine_max_iterations = 10;
static const float _affine_max_residue = 10.0;
static const float _affine_min_displacement = 0.02f;
static const float _affine_max_displacement_differ = 1.5f;

static const bool _smoothBeforeSelecting = true;
static const bool _writeInternalImages = false;
static const int _search_range = 15;
static const int _nSkippedPixels = 0;


KLT_FeatureList KLTCreateFeatureList(KLT_TrackingContext tc,int nFeatures)
{
    KLT_FeatureList fl;
    KLT_Feature first;
    int nbytes = sizeof(KLT_FeatureListRec) +
    100*nFeatures * sizeof(KLT_Feature) +
    100*nFeatures * sizeof(KLT_FeatureRec);
    int i;
    
    /* Allocate memory for feature list */
    fl = (KLT_FeatureList)malloc(nbytes);
    
    /* Set parameters */
    fl->nFeatures = nFeatures;
    
    /* Set pointers */
    fl->feature = (KLT_Feature *) (fl + 1);
    first = (KLT_Feature) (fl->feature + 100*nFeatures);
    
    for (i = 0 ; i <100*nFeatures ; i++) {
        
        fl->feature[i] = first + i;
        
        
        fl->feature[i]->aff_x=-1.0;
        fl->feature[i]->aff_y=-1.0;
        
        fl->feature[i]->aff_img=NULL;
        fl->feature[i]->aff_gradx=NULL;
        fl->feature[i]->aff_grady=NULL;
        
    }
    /* Return feature list */
    return(fl);
}



void KLTChangeTCPyramid(KLT_TrackingContext tc,
                        int search_range)
{
    float window_halfwidth;
    float subsampling;
    
    /* Check window size (and correct if necessary) */
    if (tc->window_width % 2 != 1) {
        tc->window_width = tc->window_width+1;
        KLTWarning("(KLTChangeTCPyramid) Window width must be odd.  "
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height % 2 != 1) {
        tc->window_height = tc->window_height+1;
        KLTWarning("(KLTChangeTCPyramid) Window height must be odd.  "
                   "Changing to %d.\n", tc->window_height);
    }
    if (tc->window_width < 3) {
        tc->window_width = 3;
        KLTWarning("(KLTChangeTCPyramid) Window width must be at least three.  \n"
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height < 3) {
        tc->window_height = 3;
        KLTWarning("(KLTChangeTCPyramid) Window height must be at least three.  \n"
                   "Changing to %d.\n", tc->window_height);
    }
    window_halfwidth = std::min(tc->window_width,tc->window_height)/2.0f;
    
    subsampling = ((float) search_range) / window_halfwidth;
    
    if (subsampling < 1.0)  {		/* 1.0 = 0+1 */
        tc->nPyramidLevels = 1;
    } else if (subsampling <= 3.0)  {	/* 3.0 = 2+1 */
        tc->nPyramidLevels = 2;
        tc->subsampling = 2;
    } else if (subsampling <= 5.0)  {	/* 5.0 = 4+1 */
        tc->nPyramidLevels = 2;
        tc->subsampling = 4;
    } else if (subsampling <= 9.0)  {	/* 9.0 = 8+1 */
        tc->nPyramidLevels = 2;
        tc->subsampling = 8;
    } else {
        float val = (float) (log(7.0*subsampling+1.0)/log(8.0));
        tc->nPyramidLevels = (int) (val + 0.99);
        tc->subsampling = 8;
    }
}

void computeKernels(float sigma,
                           ConvolutionKernel *gauss,
                           ConvolutionKernel *gaussderiv)
{
    const float factor = 0.01f;   /* for truncating tail */
    int i;
    
    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0);
    
    /* Compute kernels, and automatically determine widths */
    {
        const int hw = MAX_KERNEL_WIDTH / 2;
        float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
        
        /* Compute gauss and deriv */
        for (i = -hw ; i <= hw ; i++)  {
            gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
            gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
        }
        
        /* Compute widths */
        gauss->width = MAX_KERNEL_WIDTH;
        for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ;
             i++, gauss->width -= 2);
        gaussderiv->width = MAX_KERNEL_WIDTH;
        for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ;
             i++, gaussderiv->width -= 2);
        if (gauss->width == MAX_KERNEL_WIDTH ||
            gaussderiv->width == MAX_KERNEL_WIDTH)
            KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
                     "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
    }
    
    /* Shift if width less than MAX_KERNEL_WIDTH */
    
    for (i = 0 ; i < gauss->width ; i++)
        gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
    
    for (i = 0 ; i < gaussderiv->width ; i++)
        gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
    
    /* Normalize gauss and deriv */
    
    {
        const int hw = gaussderiv->width / 2;
        float den;
        
        den = 0.0;
        for (i = 0 ; i < gauss->width ; i++)
            den += gauss->data[i];
        
        for (i = 0 ; i < gauss->width ; i++)
            gauss->data[i] /= den;
        
        den = 0.0;
        for (i = -hw ; i <= hw ; i++)
            den += i*gaussderiv->data[i+hw];
        
        for (i = -hw ; i <= hw ; i++)
            gaussderiv->data[i+hw] /= den;
    }
    sigma_last = sigma;
}


/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(float sigma,
                         int *gauss_width,
                         int *gaussderiv_width){
    computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}

float _KLTComputeSmoothSigma(KLT_TrackingContext tc){
    return (tc->smooth_sigma_fact * std::max(tc->window_width, tc->window_height));
}

static float _pyramidSigma(KLT_TrackingContext tc){
    return (tc->pyramid_sigma_fact * tc->subsampling);
}



void KLTUpdateTCBorder(KLT_TrackingContext tc){
    float val;
    int pyramid_gauss_hw;
    int smooth_gauss_hw;
    int gauss_width, gaussderiv_width;
    int num_levels = tc->nPyramidLevels;
    int n_invalid_pixels;
    int window_hw;
    int ss = tc->subsampling;
    int ss_power;
    int border;
    int i;
    
    /* Check window size (and correct if necessary) */
    if (tc->window_width % 2 != 1) {
        tc->window_width = tc->window_width+1;
        KLTWarning("(KLTUpdateTCBorder) Window width must be odd.  "
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height % 2 != 1) {
        tc->window_height = tc->window_height+1;
        KLTWarning("(KLTUpdateTCBorder) Window height must be odd.  "
                   "Changing to %d.\n", tc->window_height);
    }
    if (tc->window_width < 3) {
        tc->window_width = 3;
        KLTWarning("(KLTUpdateTCBorder) Window width must be at least three.  \n"
                   "Changing to %d.\n", tc->window_width);
    }
    if (tc->window_height < 3) {
        tc->window_height = 3;
        KLTWarning("(KLTUpdateTCBorder) Window height must be at least three.  \n"
                   "Changing to %d.\n", tc->window_height);
    }
    window_hw = std::max(tc->window_width, tc->window_height)/2;
    
    /* Find widths of convolution windows */
    _KLTGetKernelWidths(_KLTComputeSmoothSigma(tc),
                        &gauss_width, &gaussderiv_width);
    
    smooth_gauss_hw = gauss_width/2;
    
    _KLTGetKernelWidths(_pyramidSigma(tc),
                        &gauss_width, &gaussderiv_width);
    
    pyramid_gauss_hw = gauss_width/2;
    
    /* Compute the # of invalid pixels at each level of the pyramid.
     n_invalid_pixels is computed with respect to the ith level
     of the pyramid.  So, e.g., if n_invalid_pixels = 5 after
     the first iteration, then there are 5 invalid pixels in
     level 1, which translated means 5*subsampling invalid pixels
     in the original level 0. */
    n_invalid_pixels = smooth_gauss_hw;
    for (i = 1 ; i < num_levels ; i++)  {
        val = ((float) n_invalid_pixels + pyramid_gauss_hw) / ss;
        n_invalid_pixels = (int) (val + 0.99);  /* Round up */
    }
    
    /* ss_power = ss^(num_levels-1) */
    ss_power = 1;
    for (i = 1 ; i < num_levels ; i++)
        ss_power *= ss;
    
    /* Compute border by translating invalid pixels back into */
    /* original image */
    border = (n_invalid_pixels + window_hw) * ss_power;
    
    tc->borderx = border;
    tc->bordery = border;
}


KLT_TrackingContext KLTCreateTrackingContext(){
    
    KLT_TrackingContext tc;
    
    /* Allocate memory */
    tc = (KLT_TrackingContext)  malloc(sizeof(KLT_TrackingContextRec));
    
    /* Set values to default values */
    tc->mindist = _mindist;
    tc->window_width = _window_size;
    tc->window_height = _window_size;
    tc->sequentialMode = _sequentialMode;
    tc->smoothBeforeSelecting = _smoothBeforeSelecting;
    tc->writeInternalImages = _writeInternalImages;
    tc->lighting_insensitive = _lighting_insensitive;
    tc->min_eigenvalue = _min_eigenvalue;
    tc->min_determinant = _min_determinant;
    tc->max_iterations = _max_iterations;
    tc->min_displacement = _min_displacement;
    tc->max_residue = _max_residue;
    tc->grad_sigma = _grad_sigma;
    tc->smooth_sigma_fact = _smooth_sigma_fact;
    tc->pyramid_sigma_fact = _pyramid_sigma_fact;
    tc->step_factor = _step_factor;
    tc->nSkippedPixels = _nSkippedPixels;
    
    
    /* for affine mapping */
    tc->affineConsistencyCheck = _affineConsistencyCheck;
    tc->affine_window_width = _affine_window_size;
    tc->affine_window_height = _affine_window_size;
    tc->affine_max_iterations = _affine_max_iterations;
    tc->affine_max_residue = _affine_max_residue;
    tc->affine_min_displacement = _affine_min_displacement;
    tc->affine_max_displacement_differ = _affine_max_displacement_differ;
    
    /* Change nPyramidLevels and subsampling */
    KLTChangeTCPyramid(tc,_search_range);
    
    /* Update border, which is dependent upon  */
    /* smooth_sigma_fact, pyramid_sigma_fact, window_size, and subsampling */
    KLTUpdateTCBorder(tc);
    
    return(tc);
}



