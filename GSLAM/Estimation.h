//
//  Optimization.h
//  GSLAM
//
//  Created by ctang on 9/4/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "KLT.h"
#include "KeyFrame.h"

namespace GSLAM {
    
    class Undistortion{
    
    };
    
    class RelativeOutlierRejection{
    public:
        Eigen::Matrix3d K;
        Eigen::Matrix3d rotation;
        float theshold;
        float prob;
        double minViewAngle;
        double medViewAngle;
        vector<bool>* isValid;
        
        int relativeRansac(const KeyFrame* keyFrame,
                           const KLT_FeatureList featureList);
    };
    
    class RelativePoseEstimation{
    public:
        
        bool rotationIsKnown;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        
        double medViewAngle;
        double minViewAngle;
        vector<bool>* isValid;
        
        int estimateRelativePose(const KeyFrame* keyFrame,
                                 const KLT_FeatureList featureList,
                                 std::vector<Eigen::Vector3d*> &pVectors);
        void unitTest();
    private:
        
    };
    
    
    class LocalFactorization{
    public:
        void process(KeyFrame *keyFrame);
        void iterativeRefine(KeyFrame *keyFrame);
    };
    
    class LocalBundleAdjustment{
        
    public:
        
        double projErrorThres;
        double viewAngleThres;
        
        void refinePoints(KeyFrame* keyFrame);
        void triangulate2(KeyFrame* keyFrame);
        
        void bundleAdjust(KeyFrame *keyFrame,bool cameraFixed=false);
        void triangulate(KeyFrame *keyFrame);
        
        void refineKeyFrameConnection(KeyFrame *keyFrame);
        
        int  refineKeyFrameMatches(KeyFrame* keyFrame1,const KeyFrame* keyFrame2,Transform& transform,
                                   std::vector<int>& matches12,std::vector<int>& matches21);
        
        //void estimateFarFrames(KeyFrame* keyFrame);
        
        int BAIterations;
    };
    
    class PnPEstimator{
    public:
        int    minCount;
        double threshold;
        double prob;
        
        int estimate(KeyFrame *keyFrame1,KeyFrame *keyFrame2,
                     const vector<int>& matches,Transform& pose);
        
        int estimate(KeyFrame *keyFrame1,KeyFrame *keyFrame2,
                     vector<int>& matches12,vector<int>& matches21,Transform& pose);
        
    };
}