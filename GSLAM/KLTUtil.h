//
//  KLTUtil.cpp
//  GSLAM
//
//  Created by ctang on 9/8/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "KLT.h"
#pragma once

void computeKernels(float sigma,ConvolutionKernel *gauss,ConvolutionKernel *gaussderiv);
float _KLTComputeSmoothSigma(KLT_TrackingContext tc);

KLT_TrackingContext KLTCreateTrackingContext();
KLT_FeatureList KLTCreateFeatureList(KLT_TrackingContext tc,int nFeatures);





