//
//  Frame.cpp
//  GSLAM
//
//  Created by ctang on 9/4/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "Frame.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace GSLAM{
    
    long unsigned int Frame::nNextId=0;
    bool Frame::mbInitialComputations=true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
    
    Frame::Frame(cv::Mat &im, const double &timeStamp,
                 ORBextractor* extractor, ORBVocabulary* voc,
                 cv::Mat &K, cv::Mat &distCoef)
    :mpORBvocabulary(voc),mpORBextractor(extractor),
    mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()){
        
        keyFramePtr=NULL;
        
        //im.copyTo(debugImage);
#ifdef  DEBUG
        //im.copyTo(debugImage);
#endif
        
        // Exctract ORB
        (*mpORBextractor)(im,cv::Mat(),mvKeys,mDescriptors);
        
        N = mvKeys.size();
        
        if(mvKeys.empty())
            return;
        
        mvpLocalMapPoints = vector<LocalMapPoint*>(N,static_cast<LocalMapPoint*>(NULL));
        
        
        UndistortKeyPoints();
        
        // This is done for the first created Frame
        if(mbInitialComputations){
            
            ComputeImageBounds(im);
            
            
            mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)
                                 /static_cast<float>(mnMaxX-mnMinX);
            
            mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)
                                 /static_cast<float>(mnMaxY-mnMinY);
            
            fx = K.at<float>(0,0);
            fy = K.at<float>(1,1);
            cx = K.at<float>(0,2);
            cy = K.at<float>(1,2);
            
            mbInitialComputations=false;
        }
        
        
        mnId=nNextId++;
        
        //Scale Levels Info
        mnScaleLevels = mpORBextractor->GetLevels();
        mfScaleFactor = mpORBextractor->GetScaleFactor();
        mfLogScaleFactor = std::log(mfScaleFactor);
        
        
        mvScaleFactors.resize(mnScaleLevels);
        mvLevelSigma2.resize(mnScaleLevels);
        
        
        mvScaleFactors[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        for(int i=1; i<mnScaleLevels; i++){
            mvScaleFactors[i]=mvScaleFactors[i-1]*mfScaleFactor;
            mvLevelSigma2[i]=mvScaleFactors[i]*mvScaleFactors[i];
        }
        
        mvInvLevelSigma2.resize(mvLevelSigma2.size());
        for(int i=0; i<mnScaleLevels; i++)
            mvInvLevelSigma2[i]=1/mvLevelSigma2[i];
        
        // Assign Features to Grid Cells
        int nReserve = 0.5*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
        
        for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
            for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
                mGrid[i][j].reserve(nReserve);
        
        
        for(size_t i=0;i<mvKeysUn.size();i++){
            cv::KeyPoint &kp = mvKeysUn[i];
            int nGridPosX, nGridPosY;
            if(PosInGrid(kp,nGridPosX,nGridPosY)){
                mGrid[nGridPosX][nGridPosY].push_back(i);
            }
        }
        
        mvbOutlier = vector<bool>(N,false);
        ComputeBoW();
    }
    
    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY){
        
        posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
        posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);
        
        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
            return false;
        
        return true;
    }
    
    void Frame::ComputeImageBounds(const cv::Mat &im)
    {
        if(mDistCoef.at<float>(0)!=0.0){
            
            cv::Mat mat(4,2,CV_32F);
            mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
            mat.at<float>(1,0)=im.cols; mat.at<float>(1,1)=0.0;
            mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=im.rows;
            mat.at<float>(3,0)=im.cols; mat.at<float>(3,1)=im.rows;
            
            // Undistort corners
            mat=mat.reshape(2);
            cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
            mat=mat.reshape(1);
            
            mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
            mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
            mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
            mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
            
        }else{
            
            mnMinX = 0.0f;
            mnMaxX = im.cols;
            mnMinY = 0.0f;
            mnMaxY = im.rows;
        }
    }
    
    void Frame::UndistortKeyPoints()
    {
        if(mDistCoef.at<float>(0)==0.0){
            mvKeysUn=mvKeys;
            return;
        }
        
        // Fill matrix with points
        cv::Mat mat(N,2,CV_32F);
        for(int i=0; i<N; i++)
        {
            mat.at<float>(i,0)=mvKeys[i].pt.x;
            mat.at<float>(i,1)=mvKeys[i].pt.y;
        }
        
        // Undistort points
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);
        
        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for(int i=0; i<N; i++)
        {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            mvKeysUn[i]=kp;
        }
    }
    
    void Frame::ComputeBoW(){
        
        if(mBowVec.empty()){
            vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        }
    }
    
    std::vector<cv::Mat> Frame::toDescriptorVector(const cv::Mat &Descriptors){
        
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);
        for (int j=0;j<Descriptors.rows;j++)
            vDesc.push_back(Descriptors.row(j));
        return vDesc;
    }
    
    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r,
                                            const int minLevel, const int maxLevel,vector<float> &distances) const
    {
        vector<size_t> vIndices;
        vIndices.reserve(N);
        distances.reserve(N);
        
        const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            return vIndices;
        
        const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            return vIndices;
        
        const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            return vIndices;
        
        const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            return vIndices;
        
        const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);
        
        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGrid[ix][iy];
                if(vCell.empty())
                    continue;
                
                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if(bCheckLevels){
                        
                        if(kpUn.octave<minLevel)
                            continue;
                        
                        if(maxLevel>=0)
                            if(kpUn.octave>maxLevel)
                                continue;
                    }
                    
                    const float distx = kpUn.pt.x-x;
                    const float disty = kpUn.pt.y-y;
                    
                    if(fabs(distx)<r && fabs(disty)<r){
                        vIndices.push_back(vCell[j]);
                        distances.push_back(distx*distx+disty*disty);
                    }
                }
            }
        }
        
        vIndices.shrink_to_fit();
        distances.shrink_to_fit();
        
        return vIndices;
    }
}
