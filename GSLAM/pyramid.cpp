//
//  pyramid.cpp
//  GSLAM
//
//  Created by ctang on 9/8/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "pyramid.h"
#include "KLTUtil.h"
#include <iostream>
#include "opencv2/core/core.hpp"
//#include "opencv2/video/tracking.hpp"

//void computePyramid(KLT_TrackingContext tc,
//                    const cv::Mat &img,
//                    std::vector<cv::Mat> &pyramid){
//    
//    if (pyramid.size()!=3*tc->nPyramidLevels) {
//        pyramid.resize(3*tc->nPyramidLevels);
//    }
//    
//    int ncols = img.cols, nrows = img.rows;
//    float pyramid_sigma = tc->subsampling * tc->pyramid_sigma_fact;  /* empirically determined */
//    
//    cv::Mat tmpimg;
//    img.convertTo(tmpimg,CV_32F);
//    
//    computeSmoothedImage(tmpimg,pyramid[0],_KLTComputeSmoothSigma(tc));
//    computeGradients(pyramid[0],pyramid[1],pyramid[2],tc->grad_sigma);
//    for (int i = 1 ; i < tc->nPyramidLevels ; i++)  {
//        computeSmoothedImage(pyramid[3*i-3],tmpimg,pyramid_sigma);
//        ncols /= tc->subsampling;  nrows /= tc->subsampling;
//        cv::resize(tmpimg,pyramid[3*i],cv::Size(ncols,nrows));
//        computeGradients(pyramid[3*i],pyramid[3*i+1],pyramid[3*i+2],tc->grad_sigma);
//    }
//}

#include <thread>
#include "KLTUtil.h"
inline void computeGxGy(const cv::Mat& img,cv::Mat& grad,cv::Mat& filter1,cv::Mat& filter2){
    cv::sepFilter2D(img,grad,CV_32F,filter1,filter2);
}

void computePyramid(KLT_TrackingContext tc,
                     const cv::Mat &img,
                     std::vector<cv::Mat> &pyramid){
    
    if (pyramid.size()!=3*tc->nPyramidLevels) {
        pyramid.resize(3*tc->nPyramidLevels);
    }
    
    
    
    int ncols = img.cols, nrows = img.rows;
    float pyramid_sigma = tc->subsampling * tc->pyramid_sigma_fact;  /* empirically determined */
    
    cv::Mat tmpimg;
    img.convertTo(tmpimg,CV_32F);
    
    computeSmoothedImage(tmpimg,pyramid[0],_KLTComputeSmoothSigma(tc));
    for (int i = 1 ; i < tc->nPyramidLevels ; i++)  {
        computeSmoothedImage(pyramid[3*i-3],tmpimg,pyramid_sigma);
        ncols /= tc->subsampling;  nrows /= tc->subsampling;
        cv::resize(tmpimg,pyramid[3*i],cv::Size(ncols,nrows));
    }
    
    
    if (fabs(tc->grad_sigma - sigma_last) > 0.05){
        computeKernels(tc->grad_sigma,&gauss_kernel, &gaussderiv_kernel);
    }
    cv::Mat kernelGauss(1,gauss_kernel.width,CV_32F,gauss_kernel.data);
    cv::Mat kernelGaussDeriv(1,gaussderiv_kernel.width,CV_32F,gaussderiv_kernel.data);
    
    std::vector<std::thread> threads(2*tc->nPyramidLevels);
    for (int i = 0 ; i < tc->nPyramidLevels ; i++)  {
        threads[2*i]=std::thread(computeGxGy,std::ref(pyramid[3*i]),std::ref(pyramid[3*i+1]),std::ref(kernelGaussDeriv),std::ref(kernelGauss));
        threads[2*i+1]=std::thread(computeGxGy,std::ref(pyramid[3*i]),std::ref(pyramid[3*i+2]),std::ref(kernelGauss),std::ref(kernelGaussDeriv));
    }
    
    for (int i=0;i<threads.size();i++) {
        threads[i].join();
    }
}

