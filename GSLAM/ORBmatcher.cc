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

#include "ORBmatcher.h"
#include <limits.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "DBoW2/DBoW2/FeatureVector.h"
#include <stdint.h>
#include <utility>
#include "Geometry.h"

using namespace std;

namespace GSLAM
{

    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 64;
    const int ORBmatcher::HISTO_LENGTH = 30;
    
    enum{
        NOT_MATCHED=-1,
        NO_MATCH_CANDIDATE=-2,
        LARGE_MATCH_DISTANCE=-3,
        OUT_OF_DISTANCE_RANGE=-4,
        ROTATION_INCONSISTENT=-5
    };
    

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri){


    }
    
    bool errorCompare(const pair<int,double>& e1, const pair<int,double>& e2) {
        return e1.second < e2.second;
    }
    
    
    int ORBmatcher::SearchByRotation(Frame* frame1,Frame* frame2,
                                     const cv::Mat&  rotation,
                                     std::vector<cv::Point2f>& pts1,
                                     std::vector<cv::Point2f>& pts2,
                                     int ORBdist){
        
        int nMatches=0;
        std::vector<int> matches12(frame1->mvKeys.size(),-1);
        std::vector<int> matches21(frame2->mvKeys.size(),-1);
        pts1.clear();
        pts2.clear();
        
        for (int p=0;p<frame1->mvKeys.size();p++) {
            
            if (matches12[p]>=0) {
                nMatches++;
                continue;
            }
            
            if (true) {
                
                cv::Mat ptMat(3,1,CV_64FC1);
                ptMat.at<double>(0)=frame1->mvKeys[p].pt.x;
                ptMat.at<double>(1)=frame1->mvKeys[p].pt.y;
                ptMat.at<double>(2)=1.0;
                ptMat=rotation*ptMat;
                
                float u=(float)(ptMat.at<double>(0)/ptMat.at<double>(2)),
                      v=(float)(ptMat.at<double>(1)/ptMat.at<double>(2));
                
                if(u<frame2->mnMinX || u>frame2->mnMaxX)
                    continue;
                
                if(v<frame2->mnMinY || v>frame2->mnMaxY)
                    continue;
                
                
                std::vector<float> squareDistances;
                
                const vector<size_t> vIndices2 = frame2->GetFeaturesInArea(u,v,20.0,0,2,squareDistances);
                
                if(vIndices2.empty()){
                    matches12[p]=NO_MATCH_CANDIDATE;
                    continue;
                }
                
                const cv::Mat dMP = frame1->mDescriptors.row(p);
                
                int bestDist = 256;
                int bestIdx2 = -1;
                
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++){
                    
                    const size_t i2 = *vit;
                    
                    if(matches21[i2]>=0){
                        continue;
                    }
                    
                    const cv::Mat &d = frame2->mDescriptors.row(i2);
                    const int dist = DescriptorDistance(dMP,d);
                    
                    if(dist<bestDist){
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }
                
                if(bestDist<=ORBdist){
                    
                    matches21[bestIdx2]=p;
                    matches12[p]=bestIdx2;
                    
                    pts1.push_back(frame1->mvKeys[p].pt);
                    pts2.push_back(frame2->mvKeys[bestIdx2].pt);
                    
                    nMatches++;
                }else{
                    matches12[p]=LARGE_MATCH_DISTANCE;
                }
            }
        }
        return nMatches;
    }
    
    int ORBmatcher::SearchByTracking(KeyFrame *keyFrame1,KeyFrame *keyFrame2,
                                     vector<int> &matches12,vector<int> &matches21,
                                     const int ORBdist){
        
        std::vector<pair<int,double> > errorIndex;
        for (int p=0;p<keyFrame1->mvLocalMapPoints.size();p++) {
            if (keyFrame1->mvLocalMapPoints[p].isEstimated
              &&keyFrame1->mvLocalMapPoints[p].isValid()
              &&keyFrame1->mvLocalMapPoints[p].vecs.back()!=NULL) {
                errorIndex.push_back(make_pair(p,keyFrame1->mvLocalMapPoints[p].errors.back()*K(0,0)));
            }
        }
        std::sort(errorIndex.begin(),errorIndex.end(),errorCompare);
        
        int nMatches=0;
        for (int i=0;i<errorIndex.size();i++) {
            
            int p=errorIndex[i].first;
            float error=(float)errorIndex[i].second;
            Eigen::Vector3d project=K*(*keyFrame1->mvLocalMapPoints[p].vecs.back());
            project/=project(2);
            
            
            const int nPredictedLevel=keyFrame1->framePtr->mvKeysUn[p].octave;
            float radius=3*std::max(error,2.f);
            
            vector<float> squareDistances;
            const vector<size_t> vIndices2 = keyFrame2->framePtr->GetFeaturesInArea(project(0),project(1),
                                                                                    radius,nPredictedLevel-2,
                                                                                    nPredictedLevel+2,
                                                                                    squareDistances);
            if (vIndices2.empty()) {
                matches12[p]=NO_MATCH_CANDIDATE;
                continue;
            }
            
            const cv::Mat dMP = keyFrame1->framePtr->mDescriptors.row(p);
            int bestDist = 256;
            int bestIdx2 = -1;
            
            for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end();vit++){
                
                const size_t i2 = *vit;
                
                if(matches21[i2]>=0){
                    continue;
                }
                
                const cv::Mat &d = keyFrame2->framePtr->mDescriptors.row(i2);
                const int dist = DescriptorDistance(dMP,d);
                
                if(dist<bestDist){
                    bestDist=dist;
                    bestIdx2=i2;
                }
            }
            
            if(bestDist<=ORBdist){
                matches12[p]=bestIdx2;
                matches21[bestIdx2]=p;
                nMatches++;
            }else{
                matches12[p]=LARGE_MATCH_DISTANCE;
            }
        }
        return nMatches;
    }
    
    int ORBmatcher::SearchByProjection(KeyFrame *keyFrame1,KeyFrame *keyFrame2,Transform transform,
                                       vector<int> &matches12,vector<int> &matches21,const float th,const int ORBdist){
        
        int nMatches=0;
        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;
        
        for (int p=0;p<keyFrame1->mvLocalMapPoints.size();p++) {
            
            if (matches12[p]>=0) {
                nMatches++;
                continue;
            }

            if (keyFrame1->mvLocalMapPoints[p].isEstimated) {
                
                Eigen::Vector3d position1=keyFrame1->mvLocalMapPoints[p].getPosition();
                Eigen::Vector3d position2=transform.rotation*(keyFrame1->mvLocalMapPoints[p].getPosition()-transform.translation);
                Eigen::Vector3d predict;
                
                predict=K*position2;
                predict/=predict(2);
                
                float u=(float)predict(0),v=(float)predict(1);
                if(u<keyFrame2->framePtr->mnMinX || u>keyFrame2->framePtr->mnMaxX)
                    continue;
                if(v<keyFrame2->framePtr->mnMinY || v>keyFrame2->framePtr->mnMaxY)
                    continue;
                
                
                Eigen::Vector3d PO= position1-transform.translation;
                double dist3D=PO.norm();
                
                const float maxDistance=keyFrame1->mvLocalMapPoints[p].GetMaxDistanceInvariance();
                const float minDistance=keyFrame1->mvLocalMapPoints[p].GetMinDistanceInvariance();
                
                if(dist3D<minDistance || dist3D>maxDistance){
                    matches12[p]=OUT_OF_DISTANCE_RANGE;
                    continue;
                }
                
                int nPredictedLevel =keyFrame1->mvLocalMapPoints[p].PredictScale(dist3D,keyFrame2->framePtr->mfLogScaleFactor);
                nPredictedLevel=nPredictedLevel<0?0:nPredictedLevel;
                nPredictedLevel=nPredictedLevel>=keyFrame1->framePtr->mnScaleLevels?
                                keyFrame1->framePtr->mnScaleLevels-1:nPredictedLevel;
                
                
                const float radius = th*keyFrame2->framePtr->mvScaleFactors[nPredictedLevel];
                
                std::vector<float> squareDistances;
                const vector<size_t> vIndices2 = keyFrame2->framePtr->GetFeaturesInArea(u,v,radius,
                                                                                        nPredictedLevel-2,nPredictedLevel+2,
                                                                                        squareDistances);
                if(vIndices2.empty()){
                    matches12[p]=NO_MATCH_CANDIDATE;
                    continue;
                }
                
                const cv::Mat dMP = keyFrame1->framePtr->mDescriptors.row(p);
                
                int bestDist = 256;
                int bestIdx2 = -1;
                
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++){
                    
                    const size_t i2 = *vit;
                    
                    if(matches21[i2]>=0){
                        continue;
                    }
                    
                    const cv::Mat &d = keyFrame2->framePtr->mDescriptors.row(i2);
                    const int dist = DescriptorDistance(dMP,d);
                    
                    if(dist<bestDist){
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist){
                    
                    matches21[bestIdx2]=p;
                    matches12[p]=bestIdx2;
                    nMatches++;
                    
                    if(mbCheckOrientation){
                        float rot = keyFrame1->framePtr->mvKeysUn[p].angle-keyFrame1->framePtr->mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }else{
                    matches12[p]=LARGE_MATCH_DISTANCE;
                }
            }
        }
        
        if(mbCheckOrientation){
            
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;
            
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            for(int i=0; i<HISTO_LENGTH; i++){
                if(i!=ind1 && i!=ind2 && i!=ind3){
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++){
                        matches12[matches21[rotHist[i][j]]]=ROTATION_INCONSISTENT;
                        matches21[rotHist[i][j]]=NOT_MATCHED;
                        nMatches--;
                    }
                }
            }
        }
        return nMatches;
    }
    
    
    int ORBmatcher::SearchByProjectionLoop(KeyFrame *keyFrame1,KeyFrame *keyFrame2,Transform transform,
                                       vector<int> &matches12,vector<int> &matches21,
                                           const float th,const int ORBdist){
        
        int nMatches=0;
        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;
        
        for (int p=0;p<keyFrame1->mvLocalMapPoints.size();p++) {
            
            if (matches12[p]>=0) {
                assert(matches21[matches12[p]]==p);
                nMatches++;
                continue;
            }
            
            if (keyFrame1->mvLocalMapPoints[p].isEstimated) {
                
                Eigen::Vector3d position1=keyFrame1->mvLocalMapPoints[p].getPosition();
                Eigen::Vector3d position2=transform.rotation*(keyFrame1->mvLocalMapPoints[p].getPosition()-transform.translation);
                Eigen::Vector3d predict;
                
                predict=K*position2;
                predict/=predict(2);
                
                float u=(float)predict(0),v=(float)predict(1);
                if(u<keyFrame2->framePtr->mnMinX || u>keyFrame2->framePtr->mnMaxX)
                    continue;
                if(v<keyFrame2->framePtr->mnMinY || v>keyFrame2->framePtr->mnMaxY)
                    continue;
                
                
                Eigen::Vector3d PO= position1-transform.translation;
                double dist3D=PO.norm();
                
                const float maxDistance=keyFrame1->mvLocalMapPoints[p].GetMaxDistanceInvariance();
                const float minDistance=keyFrame1->mvLocalMapPoints[p].GetMinDistanceInvariance();
                
                if(dist3D<minDistance || dist3D>maxDistance){
                    matches12[p]=OUT_OF_DISTANCE_RANGE;
                    continue;
                }
                
                int nPredictedLevel =keyFrame1->mvLocalMapPoints[p].PredictScale(dist3D,keyFrame2->framePtr->mfLogScaleFactor);
                nPredictedLevel=nPredictedLevel<0?0:nPredictedLevel;
                nPredictedLevel=nPredictedLevel>=keyFrame1->framePtr->mnScaleLevels?
                keyFrame1->framePtr->mnScaleLevels-1:nPredictedLevel;
                
                
                const float radius = th*keyFrame2->framePtr->mvScaleFactors[nPredictedLevel];
                
                std::vector<float> squareDistances;
                const vector<size_t> vIndices2 = keyFrame2->framePtr->GetFeaturesInArea(u,v,radius,
                                                                                        nPredictedLevel-2,nPredictedLevel+2,
                                                                                        squareDistances);
                if(vIndices2.empty()){
                    matches12[p]=NO_MATCH_CANDIDATE;
                    continue;
                }
                
                const cv::Mat dMP = keyFrame1->framePtr->mDescriptors.row(p);
                
                int bestDist = 256;
                int bestIdx2 = -1;
                
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++){
                    
                    const size_t i2 = *vit;
                    
                    if(matches21[i2]>=0){
                        continue;
                    }
                    
                    const cv::Mat &d = keyFrame2->framePtr->mDescriptors.row(i2);
                    const int dist = DescriptorDistance(dMP,d);
                    
                    if(dist<bestDist){
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }
                
                if(bestDist<=ORBdist){
                    
                    matches21[bestIdx2]=p;
                    matches12[p]=bestIdx2;
                    nMatches++;
                    
                    if(mbCheckOrientation){
                        float rot = keyFrame1->framePtr->mvKeysUn[p].angle-keyFrame1->framePtr->mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }else{
                    matches12[p]=LARGE_MATCH_DISTANCE;
                }
            }
        }
        
        if(mbCheckOrientation){
            
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;
            
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            for(int i=0; i<HISTO_LENGTH; i++){
                if(i!=ind1 && i!=ind2 && i!=ind3){
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++){
                        matches12[matches21[rotHist[i][j]]]=ROTATION_INCONSISTENT;
                        matches21[rotHist[i][j]]=NOT_MATCHED;
                        nMatches--;
                    }
                }
            }
        }
        return nMatches;
    }

    
    int ORBmatcher::RefineMatchByProjection(KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                                            Transform transform,vector<int> &matches12,vector<int> &matches21,
                                            const float th,const int ORBdist){
        int nMatches=0;
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;
        
        for (int p=0;p<keyFrame1->mvLocalMapPoints.size();p++) {
            
            if (matches12[p]>=0) {
                assert(matches21[matches12[p]]==p);
                nMatches++;
                continue;
            }
            
            if (keyFrame1->mvLocalMapPoints[p].isEstimated) {
                
                Eigen::Vector3d position1=keyFrame1->mvLocalMapPoints[p].getPosition();
                Eigen::Vector3d position2=transform.rotation*(keyFrame1->mvLocalMapPoints[p].getPosition()-transform.translation);
                Eigen::Vector3d predict;
                
                predict=K*position2;
                predict/=predict(2);
                
                float u=(float)predict(0),v=(float)predict(1);
                if(u<keyFrame2->framePtr->mnMinX || u>keyFrame2->framePtr->mnMaxX)
                    continue;
                if(v<keyFrame2->framePtr->mnMinY || v>keyFrame2->framePtr->mnMaxY)
                    continue;
                
                
                Eigen::Vector3d PO= position1-transform.translation;
                double dist3D=PO.norm();
                
                const float maxDistance=keyFrame1->mvLocalMapPoints[p].GetMaxDistanceInvariance();
                const float minDistance=keyFrame1->mvLocalMapPoints[p].GetMinDistanceInvariance();
                
                if(dist3D<minDistance || dist3D>maxDistance){
                    matches12[p]=OUT_OF_DISTANCE_RANGE;
                    continue;
                }
                
                int nPredictedLevel =keyFrame1->mvLocalMapPoints[p].PredictScale(dist3D,keyFrame2->framePtr->mfLogScaleFactor);
                nPredictedLevel=nPredictedLevel<0?0:nPredictedLevel;
                nPredictedLevel=nPredictedLevel>=keyFrame1->framePtr->mnScaleLevels?
                keyFrame1->framePtr->mnScaleLevels-1:nPredictedLevel;
                
                
                const float radius = th*keyFrame2->framePtr->mvScaleFactors[nPredictedLevel];
                
                std::vector<float> squareDistances;
                const vector<size_t> vIndices2 = keyFrame2->framePtr->GetFeaturesInArea(u,v,radius,
                                                                                        nPredictedLevel-2,
                                                                                        nPredictedLevel+2,
                                                                                        squareDistances);
                if(vIndices2.empty()){
                    matches12[p]=NO_MATCH_CANDIDATE;
                    continue;
                }
                
                const cv::Mat dMP = keyFrame1->framePtr->mDescriptors.row(p);
                
                int bestDist = 256;
                int bestIdx2 = -1;
                
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++){
                    
                    const size_t i2 = *vit;
                    if(matches21[i2]>=0||!keyFrame2->mvLocalMapPoints[i2].isEstimated){
                        continue;
                    }
                    
                    const cv::Mat &d = keyFrame2->framePtr->mDescriptors.row(i2);
                    const int dist = DescriptorDistance(dMP,d);
                    
                    if(dist<bestDist){
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }
                
                if(bestDist<=ORBdist){
                    
                    matches21[bestIdx2]=p;
                    matches12[p]=bestIdx2;
                    nMatches++;
                    
                    if(mbCheckOrientation){
                        float rot = keyFrame1->framePtr->mvKeysUn[p].angle-keyFrame1->framePtr->mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }else{
                    matches12[p]=LARGE_MATCH_DISTANCE;
                }
            }
        }
        
        if(mbCheckOrientation){
            
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;
            
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            for(int i=0; i<HISTO_LENGTH; i++){
                if(i!=ind1 && i!=ind2 && i!=ind3){
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++){
                        matches12[matches21[rotHist[i][j]]]=ROTATION_INCONSISTENT;
                        matches21[rotHist[i][j]]=NOT_MATCHED;
                        nMatches--;
                    }
                }
            }
        }
        return nMatches;
    }
    
    int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2,vector<int> &matches12,vector<int> &matches21)
    {
        const vector<cv::KeyPoint> &vKeysUn1  = pKF1->framePtr->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->framePtr->mFeatVec;
        const cv::Mat &Descriptors1 = pKF1->framePtr->mDescriptors;
        
        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->framePtr->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->framePtr->mFeatVec;
        const cv::Mat &Descriptors2 = pKF2->framePtr->mDescriptors;
        
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        
        const float factor = 1.0f/HISTO_LENGTH;
        
        int nmatches = 0;
        
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        
        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    
                    
                    if (matches12[idx1]>=0) {
                        nmatches++;
                        assert(0);
                        continue;
                    }
                    
                    if (!pKF1->mvLocalMapPoints[idx1].isEstimated) {
                        continue;
                    }
                    
                    const cv::Mat &d1 = Descriptors1.row(idx1);
                    
                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;
                    
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++){
                        
                        const size_t idx2 = f2it->second[i2];
                        
                        if (matches21[idx2]>=0) {
                            continue;
                        }
                        
                        const cv::Mat &d2 = Descriptors2.row(idx2);
                        
                        int dist = DescriptorDistance(d1,d2);
                        
                        if(dist<bestDist1){
                            
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                            
                        }else if(dist<bestDist2){
                            
                            bestDist2=dist;
                        }
                    }
                    
                    if(bestDist1<TH_LOW){
                        
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2)){
                            
                            matches12[idx1]=bestIdx2;
                            matches21[bestIdx2]=idx1;
                            
                            if(mbCheckOrientation){
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }
                
                f1it++;
                f2it++;
                
            }else if(f1it->first < f2it->first){
                f1it = vFeatVec1.lower_bound(f2it->first);
            }else{
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }
        
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;
            
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            
            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++){
                    matches21[matches12[rotHist[i][j]]]=-1;
                    matches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }
        }
        return nmatches;
    }
    
    
    int ORBmatcher::SearchByBoWLoop(KeyFrame *pKF1, KeyFrame *pKF2,vector<int> &matches12,vector<int> &matches21){
        
        const vector<cv::KeyPoint> &vKeysUn1  = pKF1->framePtr->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->framePtr->mFeatVec;
        const cv::Mat &Descriptors1 = pKF1->framePtr->mDescriptors;
        
        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->framePtr->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->framePtr->mFeatVec;
        const cv::Mat &Descriptors2 = pKF2->framePtr->mDescriptors;
        
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        
        const float factor = 1.0f/HISTO_LENGTH;
        
        int nmatches = 0;
        
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        
        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    
                    if (matches12[idx1]>=0) {
                        nmatches++;
                        continue;
                    }
                    
                    if (!pKF1->mvLocalMapPoints[idx1].isEstimated) {
                        continue;
                    }
                    
                    const cv::Mat &d1 = Descriptors1.row(idx1);
                    
                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;
                    
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++){
                        
                        const size_t idx2 = f2it->second[i2];
                        
                        if (matches21[idx2]>=0) {
                            continue;
                        }
                        
                        const cv::Mat &d2 = Descriptors2.row(idx2);
                        
                        int dist = DescriptorDistance(d1,d2);
                        
                        if(dist<bestDist1){
                            
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                            
                        }else if(dist<bestDist2){
                            
                            bestDist2=dist;
                        }
                    }
                    
                    if(bestDist1<TH_LOW){
                        
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2)){
                            
                            matches12[idx1]=bestIdx2;
                            matches21[bestIdx2]=idx1;
                            
                            if(mbCheckOrientation){
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }
                
                f1it++;
                f2it++;
                
            }else if(f1it->first < f2it->first){
                f1it = vFeatVec1.lower_bound(f2it->first);
            }else{
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }
        
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;
            
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            
            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++){
                    matches21[matches12[rotHist[i][j]]]=-1;
                    matches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }
        }
        return nmatches;
    }

    

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
