//
//  KLTCommon.hpp
//  STARCK
//
//  Created by Chaos on 3/31/16.
//  Copyright © 2016 Chaos. All rights reserved.
//

#include "opencv2/core/core.hpp"
#include "Eigen/Dense"
#include <vector>

#pragma once

#define USE_TBB
#define SSE_TRACKING

typedef struct  {
    
    float lighting_alpha;
    float lighting_beta;
    
    /* Available to user */
    int mindist;			/* min distance b/w features */
    int window_width, window_height;
    bool sequentialMode;	/* whether to save most recent image to save time */
    /* can set to TRUE manually, but don't set to */
    /* FALSE manually */
    bool smoothBeforeSelecting;	/* whether to smooth image before */
    /* selecting features */
    bool writeInternalImages;	/* whether to write internal images */
    /* tracking features */
    bool lighting_insensitive;  /* whether to normalize for gain and bias (not in original algorithm) */
    
    /* Available, but hopefully can ignore */
    int min_eigenvalue;		/* smallest eigenvalue allowed for selecting */
    float min_determinant;	/* th for determining lost */
    float min_displacement;	/* th for stopping tracking when pixel changes little */
    int max_iterations;		/* th for stopping tracking when too many iterations */
    float max_residue;		/* th for stopping tracking when residue is large */
    float grad_sigma;
    float smooth_sigma_fact;
    float pyramid_sigma_fact;
    float step_factor;  /* size of Newton steps; 2.0 comes from equations, 1.0 seems to avoid overshooting */
    int nSkippedPixels;		/* # of pixels skipped when finding features */
    int borderx;			/* border in which features will not be found */
    int bordery;
    int nPyramidLevels;		/* computed from search_ranges */
    int featureSelectionLevel;
    int subsampling;		/* 		" */
    
    
    /* for affine mapping */
    int affine_window_width, affine_window_height;
    int affineConsistencyCheck; /* whether to evaluates the consistency of features with affine mapping
                                 -1 = don't evaluates the consistency
                                 0 = evaluates the consistency of features with translation mapping
                                 1 = evaluates the consistency of features with similarity mapping
                                 2 = evaluates the consistency of features with affine mapping
                                 */
    int affine_max_iterations;
    float affine_max_residue;
    float affine_min_displacement;
    float affine_max_displacement_differ; /* th for the difference between the displacement calculated
                                           by the affine tracker and the frame to frame tracker in pel*/
    
}  KLT_TrackingContextRec, *KLT_TrackingContext;

#define KLT_TRACKED           0
#define KLT_NOT_FOUND        -1
#define KLT_SMALL_DET        -2
#define KLT_MAX_ITERATIONS   -3
#define KLT_OOB              -4
#define KLT_LARGE_RESIDUE    -5

typedef struct  {
    
    float x;
    float y;
    int   val;
    
    cv::Point2f pt;
    Eigen::Vector3d norm;
    Eigen::Vector3d vec;
    
    float aff_x;
    float aff_y;
    float aff_Axx;
    float aff_Ayx;
    float aff_Axy;
    float aff_Ayy;
    
    cv::Mat *aff_img;
    cv::Mat *aff_gradx;
    cv::Mat *aff_grady;
    cv::Mat coefficient;
    
}  KLT_FeatureRec,*KLT_Feature;

typedef struct  {
    bool isOutlierRejected;
    int nFeatures;
    KLT_Feature *feature;
}  KLT_FeatureListRec, *KLT_FeatureList;

/* Kernels */
#define MAX_KERNEL_WIDTH    71

typedef struct  {
    int width;
    float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

static int KLT_verbose = 0;
typedef float KLT_locType;
typedef unsigned char KLT_PixelType;



