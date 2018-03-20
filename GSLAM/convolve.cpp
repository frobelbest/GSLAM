//
//  convolve.cpp
//  GSLAM
//
//  Created by ctang on 9/8/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "convolve.h"
#include "KLTUtil.h"
#include <avxintrin.h>
#include "tbb/tbb_thread.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"

inline void computeGxGy(const cv::Mat& img,cv::Mat& grad,cv::Mat& filter1,cv::Mat& filter2){
    cv::sepFilter2D(img,grad,CV_32F,filter1,filter2);
}

void computeGradients(const cv::Mat &img,
                      cv::Mat &gradx,
                      cv::Mat &grady,
                      float sigma,
                      std::thread* threads){
    
    /* Compute kernels, if necessary */
    if (fabs(sigma - sigma_last) > 0.05){
        computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    }
    //std::cout<<gauss_kernel.width<<' '<<gaussderiv_kernel.width<<std::endl;
    //getchar();
    
    cv::Mat kernelGauss(1,gauss_kernel.width,CV_32F,gauss_kernel.data);
    cv::Mat kernelGaussDeriv(1,gaussderiv_kernel.width,CV_32F,gaussderiv_kernel.data);
    
    if (threads==NULL) {
        cv::sepFilter2D(img,gradx,CV_32F,kernelGaussDeriv,kernelGauss);
        cv::sepFilter2D(img,grady,CV_32F,kernelGauss,kernelGaussDeriv);
    }else{
        threads[0]=std::thread(computeGxGy,std::ref(img),std::ref(gradx),std::ref(kernelGaussDeriv),std::ref(kernelGauss));
        threads[1]=std::thread(computeGxGy,std::ref(img),std::ref(grady),std::ref(kernelGauss),std::ref(kernelGaussDeriv));
    }
}



void computeSmoothedImage(const cv::Mat &img,
                          cv::Mat &dst,
                          float sigma){
    /* Compute kernel, if necessary; gauss_deriv is not used */
    if (fabs(sigma - sigma_last) > 0.05){
        computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    }
    
    cv::Mat kernel(1,gauss_kernel.width,CV_32F,gauss_kernel.data);
    cv::sepFilter2D(img,dst,-1,kernel,kernel);
}



/*void sperableFilter2D(cv::Mat& img,cv::Mat& output,cv::Mat& rowFilter,cv::Mat& colFilter){
    
}

void computeGradients2(cv::Mat &img,
                       cv::Mat &gradx,
                       cv::Mat &grady,
                       float sigma){
    
    if (fabs(sigma - sigma_last) > 0.05){
        computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    }
    cv::Mat kernelGauss(1,gauss_kernel.width,CV_32F,gauss_kernel.data);
    cv::Mat kernelGaussDeriv(1,gaussderiv_kernel.width,CV_32F,gaussderiv_kernel.data);
    

    
    sperableFilter2D(img,gradx,kernelGaussDeriv,kernelGauss);
    sperableFilter2D(img,grady,kernelGauss,kernelGaussDeriv);
}



void computeSmoothedImage2(cv::Mat &img,
                           cv::Mat &dst,
                           float sigma){
    
    if (fabs(sigma - sigma_last) > 0.05){
        computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    }
    cv::Mat kernel(1,gauss_kernel.width,CV_32F,gauss_kernel.data);
    sperableFilter2D(img,dst,kernel,kernel);
}*/


