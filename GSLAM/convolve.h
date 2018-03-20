//
//  convolve.hpp
//  STARCK
//
//  Created by Chaos on 3/29/16.
//  Copyright © 2016 Chaos. All rights reserved.
//


#include "KLT.h"
#include <thread>
void computeGradients(const cv::Mat &img,cv::Mat &gradx,cv::Mat &grady,float sigma,std::thread* threads=NULL);
void computeSmoothedImage(const cv::Mat &img,cv::Mat &dst,float sigma);



//void computeGradients2(cv::Mat &img,cv::Mat &gradx,cv::Mat &grady,float sigma);
//void computeSmoothedImage2(cv::Mat &img,cv::Mat &dst,float sigma);






