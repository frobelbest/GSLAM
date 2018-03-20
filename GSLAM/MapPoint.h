//
//  Header.h
//  GSLAM
//
//  Created by ctang on 9/4/16.
//  Copyright © 2016 ctang. All rights reserved.
//
#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"

#include <vector>
#include <map>

namespace  GSLAM{
    
    class KeyFrame;
    

    class LocalMapPoint{
        
    public:
        
        Eigen::Vector3d globalPosition;
        unsigned char color[3];
        unsigned char vcolor[3];
        
        Eigen::Vector3d vec;
        Eigen::Vector3d norm;
        double          invdepth;
        
        std::vector<double>           errors;
        std::vector<Eigen::Vector3d*> vecs;
        std::vector<Eigen::Vector3d*> pVectors;
        std::vector<cv::Point2f>      tracked;
        
        bool isDeleted;
        bool isFullTrack;
        bool isEstimated;
        int measurementCount;
        
        float mfMinDistance;
        float minLevelScaleFactor;
        
        float mfMaxDistance;
        float maxLevelScaleFactor;
        
        Eigen::Vector3d getPosition() const{
            return vec/invdepth;
        }
        
        bool isValid(){
            return measurementCount>0;
        }
        
        int PredictScale(const float &currentDist,const float &logScaleFactor){
            float ratio= mfMaxDistance/currentDist;
            return int(std::ceil(std::log(ratio)/logScaleFactor));
        }
        
        float GetMinDistanceInvariance(){
            return 0.8f*mfMinDistance;
        }
        
        float GetMaxDistanceInvariance(){
            return 1.2f*mfMaxDistance;
        }
        
        void updateNormalAndDepth(){
            const float dist = (float)vec.norm()/invdepth;
            mfMaxDistance = dist*maxLevelScaleFactor;
            mfMinDistance = mfMaxDistance/minLevelScaleFactor;
        }
        
        int globalIndex;
    };
    
}
#endif
