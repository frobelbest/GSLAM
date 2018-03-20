/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include"KeyFrame.h"
#include"Frame.h"


namespace GSLAM
{

    class ORBmatcher{
    public:

        ORBmatcher(float nnratio=0.6, bool checkOri=true);

        // Computes the Hamming distance between two ORB descriptors
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        
        int SearchByBoW(KeyFrame *keyFrame1,KeyFrame *keyFrame2,std::vector<int> &matches12,std::vector<int> &matches21);
        
        int SearchByBoWLoop(KeyFrame *keyFrame1,KeyFrame *keyFrame2,std::vector<int> &matches12,std::vector<int> &matches21);
        
        int SearchByTracking(KeyFrame *keyFrame1,KeyFrame *keyFrame2,
                             std::vector<int> &matches12,std::vector<int> &matches21,int ORBdist);
        
        int SearchByProjection(KeyFrame *keyFrame1,KeyFrame *keyFrame2,Transform transform,
                              std::vector<int> &matches12,std::vector<int> &matches21,
                               const float th,const int ORBdist);
        
        int SearchByProjectionLoop(KeyFrame *keyFrame1,KeyFrame *keyFrame2,Transform transform,
                                   std::vector<int> &matches12,std::vector<int> &matches21,
                                   const float th,const int ORBdist);
        
        int SearchByRotation(Frame* frame1,Frame* frame2,const cv::Mat& rotation,
                             std::vector<cv::Point2f>& pts1,std::vector<cv::Point2f>& pts2,int ORBdist);
        
        //backward projection;
        int RefineMatchByProjection(KeyFrame *keyFrame1,KeyFrame *keyFrame2,Transform transform,
                                    std::vector<int> &matches12,std::vector<int> &matches21,
                                    const float th,const int ORBdist);
        
        
        Eigen::Matrix3d K;
        double projErrorThreshold;
        
    public:
        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;


    protected:

        bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);
        float RadiusByViewingCos(const float &viewCos);
        void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
        float mfNNratio;
        bool mbCheckOrientation;
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
