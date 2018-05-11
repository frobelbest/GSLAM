//
//  CameraPose.h
//  GSLAM
//
//  Created by ctang on 9/5/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Eigen/Dense"

namespace GSLAM{
    
    class Transform{
        
    public:
        
        Transform();
        double          scale;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        Eigen::Matrix3d E;
        
        void fromCameraToWorld();
        void fromWorldToCamera();
        void setEssentialMatrix();
        
        Transform leftMultiply(const Transform& transform)const;
        Transform inverse()const;
        
        void display()const;
    };
    
}
#endif


