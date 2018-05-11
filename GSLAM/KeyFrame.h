/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/GSLAM>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef KEYFRAME_H
#define KEYFRAME_H


#include "./DBoW2/BowVector.h"
#include "./DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "Frame.h"
#include "KLT.h"
#include "MapPoint.h"
#include "KeyFrameDatabase.h"
#include <thread>
#include <mutex>

namespace GSLAM{
    
    class KeyFrameDatabase;
    
    class LocalFrame{
    public:
        int measurementCount;
        int frameId;
        Transform pose;
    };
    
    class KeyFrame{
        
    public:
        
        std::thread baThread;
        
        KeyFrame(Frame *F,KeyFrameDatabase* pKFDB,const Eigen::Matrix3d &invK,double mScaleFactor);
        KeyFrame(Frame *F,KeyFrameDatabase* pKFDB,const Eigen::Matrix3d &invK,KLT_FeatureList featureList);
        
        KeyFrame* prevKeyFramePtr;
        KeyFrame* nextKeyFramePtr;
        
        Frame* framePtr;
        
        //Keyframe Index
        static long unsigned int nNextId;
        int mnId;
        int frameId;
        int outId;
        
        
        // MapPoints associated to keypoints
        KLT_FeatureList              mFeatureList;
        std::vector<LocalFrame>      mvLocalFrames;
        std::vector<LocalMapPoint>   mvLocalMapPoints;
        std::vector<bool>            mvRelativeEstimated;
        
        std::vector<std::vector<cv::Point2f> > trackedPoints;
        
        // BoW
        KeyFrameDatabase* mpKeyFrameDB;
        ORBVocabulary* mpORBvocabulary;
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;
        
        float mScaleFactor;
        Eigen::Matrix3d invK;
        Eigen::Matrix3d frameK;
        
        // Variables used by the keyframe database
        long unsigned int mnLoopQuery;
        int mnLoopWords;
        float mLoopScore;
        long unsigned int mnRelocQuery;
        int mnRelocWords;
        float mRelocScore;
        float minScore;
        
        
        // Covisibility
        std::set<KeyFrame *> GetConnectedKeyFrames();
        std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
        
        std::map<KeyFrame*,Transform>    mConnectedKeyFramePoses;
        std::map<KeyFrame*,vector<int> > mConnectedKeyFrameMatches;
        std::map<KeyFrame*,int>          mConnectedKeyFrameWeights;
        
        std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
        std::vector<int> mvOrderedWeights;
        
        // Global graph property
        Transform pose;
        double    scale;
        double    logScale;
        bool      isGlobalFixed;
        
        // mutex
        std::mutex mMutexPose;
        std::mutex mMutexConnections;
        std::mutex mMutexFeatures;
        
        
        //return full track count
        int appendRelativeEstimation(const int frameId,KeyFrame* keyFrame,
                                     const Eigen::Matrix3d &rotation,const Eigen::Vector3d &translation,
                                     const KLT_FeatureList featureList,const vector<Eigen::Vector3d*> &pVectors);
        
        int appendKeyFrame(KeyFrame* nextKeyFrame,Transform transform,vector<int>& matches);
        
        void savePly(const char* name);
        void saveData2(const char* name);
        void visualize();
        
    private:
        void UpdateBestCovisibles();
    };
}
#endif
