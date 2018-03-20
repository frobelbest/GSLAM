//
//  pyramid.h
//  GSLAM
//
//  Created by ctang on 9/8/16.
//  Copyright © 2016 ctang. All rights reserved.
//
#include "convolve.h"
#include "opencv2/imgproc/imgproc.hpp"

void computePyramid(KLT_TrackingContext tc,const cv::Mat &img,std::vector<cv::Mat> &pyramid);
void computePyramid3(KLT_TrackingContext tc,const cv::Mat &img,std::vector<cv::Mat> &pyramid);

void computePyramid2(KLT_TrackingContext tc,cv::Mat &img,std::vector<cv::Mat> &pyramid);

