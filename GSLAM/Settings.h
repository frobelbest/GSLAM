//
//  Common.h
//  GSLAM
//
//  Created by ctang on 9/4/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#ifndef SETTINGS_H
#define SETTINGS_H
#pragma once

#include "Eigen/Dense"
#include "IMU.hpp"

namespace GSLAM{
    
    typedef struct{
    
        double ts;
        double td;
    
        double fx;
        double fy;
        double ox;
        double oy;
    
        double k1;
        double k2;
        double p1;
        double p2;
        double k3;
    
    }CameraSettings;

    //static CameraSettings cameraSettings;

    typedef struct{
        int nFeatures;
        int nLevel;
        float scaleFactor;
    }ORBExtractorSettings;

    //static ORBExtractorSettings orbExtractorSettings;
    
    
    typedef struct{
    
    }KLTSettings;
    
    //static KLTSettings kltSettings;
    
    typedef struct{
        double medViewAngle;
        double minViewAngle;
        int    requireNewKeyFrameCount;
    }SLAMSettings;
    
    //static SLAMSettings slamSettings;
    
    //static Eigen::Matrix3d K,invK;
    //static IMU imu;
}

#define DEBUG

#endif

