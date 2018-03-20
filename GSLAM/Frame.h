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
#ifndef FRAME_H
#define FRAME_H

#include<vector>
#include "Settings.h"
#include "Transform.h"
#include "KeyFrame.h"
#include "DBoW2/DBoW2/BowVector.h"
#include "DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>


namespace GSLAM
{
    
#define FRAME_GRID_ROWS 54
#define FRAME_GRID_COLS 96

class LocalMapPoint;
class KeyFrame;
    
class Frame
{
public:
    
    Frame(cv::Mat &im, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef);
    // Copy constructor.
    Frame(const Frame &frame);

public:
    
    int mnId;
    KeyFrame* keyFramePtr;
    
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;
    
    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractor;

    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    
    int N;
    int NCoarse;
    
    cv::Mat mDescriptors;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<bool> mvbOutlier;
    std::vector<LocalMapPoint*> mvpLocalMapPoints;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    
    // Frame timestamp.
    double mTimeStamp;
    
    //static members
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;
    static bool mbInitialComputations;
    static long unsigned int nNextId;
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;
    
    
    float mfOriginalScaleFacotr;
    
    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;
    
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    void ComputeImageBounds(const cv::Mat &im);
    void UndistortKeyPoints();
    void ComputeBoW();
    
    vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
    vector<size_t>  GetFeaturesInArea(const float &x, const float  &y, const float  &r,
                                      const int minLevel, const int maxLevel,vector<float> &distances) const;
    

#ifdef DEBUG
    cv::Mat debugImage;
#endif

};

}
#endif


