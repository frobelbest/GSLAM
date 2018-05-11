//
//  System.cpp
//  GSLAM
//
//  Created by ctang on 9/3/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "System.h"
#include "KLTUtil.h"
#include "pyramid.h"
#include "LK.hpp"
#include "Estimation.h"
#include "selectGoodFeatures.hpp"
#include <iostream>
#include <tbb/tbb.h>


namespace GSLAM{
    
    System::System(const string &strVocFile, const string &strSettingsFile){
        
        // Output welcome message
        cout << endl <<
        "GSLAM Copyright (C) 2014-2016 Chengzhou Tang, Simon Fraser University." << endl <<
        "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
        "This is free software, and you are welcome to redistribute it" << endl <<
        "under certain conditions. See LICENSE.txt." << endl << endl;
        
        
        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }
        
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        
        mpVocabulary = new ORBVocabulary();
        
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;
        
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
        cout << "Keyframe Database created!" << endl << endl;
        
        
        //loading parameter
        cameraSettings.fx=fsSettings["Camera.fx"];
        cameraSettings.fy=fsSettings["Camera.fy"];
        cameraSettings.ox=fsSettings["Camera.ox"];
        cameraSettings.oy=fsSettings["Camera.oy"];
        
        mK=cv::Mat::zeros(3,3,CV_64FC1);
        mK.at<double>(0,0) = cameraSettings.fx;
        mK.at<double>(1,1) = cameraSettings.fy;
        mK.at<double>(0,2) = cameraSettings.ox;
        mK.at<double>(1,2) = cameraSettings.oy;
        
        
        cameraSettings.k1=fsSettings["Camera.k1"];
        cameraSettings.k2=fsSettings["Camera.k2"];
        cameraSettings.p1=fsSettings["Camera.p1"];
        cameraSettings.p2=fsSettings["Camera.p2"];
        cameraSettings.k3=fsSettings["Camera.k3"];

        mDistCoef=cv::Mat(4,1,CV_32F);
        mDistCoef.at<float>(0) = fsSettings["Camera.k1"];
        mDistCoef.at<float>(1) = fsSettings["Camera.k2"];
        mDistCoef.at<float>(2) = fsSettings["Camera.p1"];
        mDistCoef.at<float>(3) = fsSettings["Camera.p2"];
        
        const float k3 = fsSettings["Camera.k3"];
        if(k3!=0){
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = k3;
        }
        
        int     nFeatures   = fsSettings["ORBextractor.nFeatures"];
        float   fScaleFactor= fsSettings["ORBextractor.scaleFactor"];
        int     nLevels     = fsSettings["ORBextractor.nLevels"];
        int     fIniThFAST  = fsSettings["ORBextractor.iniThFAST"];
        int     fMinThFAST  = fsSettings["ORBextractor.minThFAST"];
        
        mpORBExtractor= new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
        mpORBExtractorFrame= new ORBextractor(1000,1.2,2,fIniThFAST,fMinThFAST);
        
        //orbMatcher=new ORBmatcher(0.8,false);
        trackingContext=KLTCreateTrackingContext();
        
        
        trackingContext->window_width=fsSettings["KLT.window_width"];
        trackingContext->window_height=fsSettings["KLT.window_height"];
        
        trackingContext->nPyramidLevels=fsSettings["KLT.nPyramidLevels"];
        trackingContext->borderx=fsSettings["KLT.borderx"];
        trackingContext->bordery=fsSettings["KLT.bordery"];
        
        trackingContext->min_eigenvalue=fsSettings["KLT.min_eigenvalue"];
        trackingContext->mindist=fsSettings["KLT.mindist"];
        
        mScaleFactor=std::pow(2,trackingContext->nPyramidLevels-1);
        mK=mK/mScaleFactor;
        mK.at<double>(2,2)=1.0;
        mKInv=mK.inv();
        
        
        featureList=KLTCreateFeatureList(trackingContext,2000);
        
        
        frameId=0;
        
        K=Eigen::Matrix3d::Zero();
        K(0,0)=cameraSettings.fx;
        K(1,1)=cameraSettings.fy;
        K(2,2)=1.0;
        K(0,2)=cameraSettings.ox;
        K(1,2)=cameraSettings.oy;
        invK=K.inverse();
        
        cvInvK=cv::Mat(3,3,CV_64FC1);
        for (int r1=0;r1<3;r1++) {
            for (int r2=0;r2<3;r2++) {
                cvInvK.at<double>(r1,r2)=invK(r1,r2);
            }
        }
        
        
        imu.ts=fsSettings["Camera.ts"];
        cameraSettings.td=fsSettings["Camera.td"];
        
        slamSettings.medViewAngle=fsSettings["SLAM.medViewAngle"];
        slamSettings.medViewAngle=std::cos((slamSettings.medViewAngle/180.0)*CV_PI);
        slamSettings.minViewAngle=fsSettings["SLAM.minViewAngle"];
        slamSettings.minViewAngle=std::cos((slamSettings.minViewAngle/180.0)*CV_PI);
        
        
        slamSettings.requireNewKeyFrameCount=fsSettings["SLAM.requireNewKeyFrameCount"];
        
        keyFrameConnector.matcherByTracking=new ORBmatcher(0.8,false);
        keyFrameConnector.matcherByProjection=new ORBmatcher(0.8,false);
        keyFrameConnector.matcherByBoW=new ORBmatcher(0.75,true);
        
        keyFrameConnector.matcherByTracking->K=K/mScaleFactor;
        keyFrameConnector.matcherByTracking->K(2,2)=1.0;
        
        keyFrameConnector.matcherByProjection->K=keyFrameConnector.matcherByTracking->K;
        
        
        lightFramePre=NULL;
        lightFrameCur=NULL;
        
        frameStart=fsSettings["SLAM.startFrame"];
        frameEnd=fsSettings["SLAM.endFrame"];
    }
    
    void processKeyFrame(System& system,KeyFrame& keyFrame){
        
        LocalFactorization localFactorization;
        localFactorization.process(&keyFrame);
        //localFactorization.iterativeRefine(keyFrame);
        
#ifdef DEBUG
        //sprintf(name,"/Users/chaos/Desktop/debug/init_%d_%d.ply",frontKeyFrame->frameId,frameId);
        //keyFrame->savePly(name);
#endif
        
        LocalBundleAdjustment localBundleAdjustment;
        localBundleAdjustment.projErrorThres=0.005;
        localBundleAdjustment.viewAngleThres=system.slamSettings.minViewAngle;
        //printf("ba\n");
        
        localBundleAdjustment.bundleAdjust(&keyFrame);
        
        
        //sprintf(name,"/Users/chaos/Desktop/debug/ba0_%d_%d.ply",frontKeyFrame->frameId,frameId);
        //keyFrame->savePly(name);
        
        localBundleAdjustment.triangulate2(&keyFrame);
        
        //sprintf(name,"/Users/chaos/Desktop/debug/tra_%d_%d.ply",frontKeyFrame->frameId,frameId);
        //keyFrame->savePly(name);
        
        
        localBundleAdjustment.bundleAdjust(&keyFrame);
        //localBundleAdjustment.refinePoints(&keyFrame);
        
        //keyFrame->visualize();
        
        //localBundleAdjustment.estimateCloseFrames(keyFrame);
        //getchar();
#ifdef DEBUG
        char name[200];
        //sprintf(name,"/Users/chaos/Desktop/syndata/data_%d.txt",keyFrame.frameId);
        //keyFrame.saveData2(name);
        sprintf(name,"/Users/chaos/Desktop/debug/data_%d.ply",keyFrame.frameId);
        keyFrame.savePly(name);
#endif
        if(keyFrame.prevKeyFramePtr!=NULL){
            if (keyFrame.prevKeyFramePtr->baThread.joinable()) {
                keyFrame.prevKeyFramePtr->baThread.join();
            }
            
            system.keyFrameConnector.mpORBVocabulary=system.mpVocabulary;
            system.keyFrameConnector.keyFrameDatabase=system.mpKeyFrameDatabase;
            system.keyFrameConnector.connectionThreshold=80;
            system.keyFrameConnector.connectKeyFrame(keyFrame.prevKeyFramePtr,&keyFrame);
        }
        
        system.mpKeyFrameDatabase->add(&keyFrame);
        system.globalReconstruction.addNewKeyFrame(&keyFrame);
    }
    
    class NormalizeFrameInvoker{
    private:
        KeyFrame* keyFrame;
        ImageGrids* grids;
        KLT_FeatureList* featureList;
        std::vector<cv::Mat>* rotations;
        cv::Mat* cvInvK;
    public:
        NormalizeFrameInvoker(KeyFrame* _keyFrame,
                              ImageGrids* _grids,
                              KLT_FeatureList* _featureList,
                              std::vector<cv::Mat>* _rotations,
                              cv::Mat* _cvInvK){
            keyFrame=_keyFrame;
            grids=_grids;
            featureList=_featureList;
            rotations=_rotations;
            cvInvK=_cvInvK;
        }
        void operator ()(const tbb::blocked_range<size_t>& range) const;
    };
    
    
    void NormalizeFrameInvoker::operator()(const tbb::blocked_range<size_t>& range) const{
        
        for (int i=range.begin();i<range.end();i++) {
            
            if ((*featureList)->feature[i]->val==KLT_TRACKED) {
                
                cv::Mat dst;
                grids->rotateAndNormalizePoint(cv::Point2f((*featureList)->feature[i]->x,
                                                           (*featureList)->feature[i]->y),
                                               dst,*rotations,*cvInvK);
                
                (*featureList)->feature[i]->norm(0)=dst.at<double>(0);
                (*featureList)->feature[i]->norm(1)=dst.at<double>(1);
                (*featureList)->feature[i]->norm(2)=dst.at<double>(2);
                (*featureList)->feature[i]->vec=(*featureList)->feature[i]->norm
                                                /(*featureList)->feature[i]->norm(2);
                
                keyFrame->mvLocalMapPoints[i].tracked.push_back(cv::Point2f((*featureList)->feature[i]->x,
                                                                            (*featureList)->feature[i]->y));
            }
        }
    }
    
    class NormalizeKeyFrameInvoker{
        
    private:
        KeyFrame* keyFrame;
        ImageGrids* grids;
        KLT_FeatureList* featureList;
        std::vector<cv::Mat>* rotations;
        cv::Mat* cvInvK;
        cv::Mat* colorImage;
    public:
        NormalizeKeyFrameInvoker(KeyFrame* _keyFrame,
                              ImageGrids* _grids,
                              KLT_FeatureList* _featureList,
                              std::vector<cv::Mat>* _rotations,
                              cv::Mat* _cvInvK,
                              cv::Mat* _colorImage){
            keyFrame=_keyFrame;
            grids=_grids;
            featureList=_featureList;
            rotations=_rotations;
            cvInvK=_cvInvK;
            colorImage=_colorImage;
        }

   
        void operator ()(const tbb::blocked_range<size_t>& range) const;
    };
    
    void NormalizeKeyFrameInvoker::operator()(const tbb::blocked_range<size_t>& range) const{
        for(int i=range.begin();i<range.end();i++){
            
            cv::Mat dst;
            grids->rotateAndNormalizePoint(cv::Point2f((*featureList)->feature[i]->x,(*featureList)->feature[i]->y),
                                           dst,*rotations,*cvInvK);
            
            keyFrame->mvLocalMapPoints[i].norm(0)=dst.at<double>(0);
            keyFrame->mvLocalMapPoints[i].norm(1)=dst.at<double>(1);
            keyFrame->mvLocalMapPoints[i].norm(2)=dst.at<double>(2);
            keyFrame->mvLocalMapPoints[i].vec=keyFrame->mvLocalMapPoints[i].norm
                                             /keyFrame->mvLocalMapPoints[i].norm(2);
            
            cv::Vec3b color=colorImage->at<cv::Vec3b>(round((*featureList)->feature[i]->y/2),
                                                      round((*featureList)->feature[i]->x/2));
            
            keyFrame->mvLocalMapPoints[i].color[0]=color.val[0];
            keyFrame->mvLocalMapPoints[i].color[1]=color.val[1];
            keyFrame->mvLocalMapPoints[i].color[2]=color.val[2];
            keyFrame->mvLocalMapPoints[i].tracked.push_back(cv::Point2f((*featureList)->feature[i]->x,
                                                                        (*featureList)->feature[i]->y));
        }
    }
    
    
    
    Transform System::Track(cv::Mat &im, const double &timestamp,const int outIndex){
        Transform pose;
        
#ifdef DEBUG
        
        static vector<cv::Scalar> colors;
        static vector<cv::Point2f> keyPoints;

        trackingContext->min_eigenvalue=100;
        trackingContext->mindist=10;
        
        trackingContext->writeInternalImages = FALSE;
        trackingContext->affineConsistencyCheck = FALSE;
        trackingContext->lighting_insensitive=FALSE;
        
        trackingContext->window_width=21;
        trackingContext->window_height=21;
        trackingContext->nPyramidLevels=2;
        trackingContext->borderx=50;
        trackingContext->bordery=50;
        
#endif
        
        if (Frame::mbInitialComputations) {
            //google::InitGoogleLogging("test");
            //for get rotation need to change!
            int frameWidth=im.cols,frameHeight=im.rows;
            frameSize=cv::Size(frameWidth,frameHeight);
            cv::Size gridSize(20,20),sizeByGrid;
            sizeByGrid.width=frameWidth/gridSize.width;
            sizeByGrid.height=frameHeight/gridSize.height;
            sizeByVertex=sizeByGrid+cv::Size(1,1);
            grids.initialize(gridSize,sizeByGrid,sizeByVertex);
            rotations.resize(sizeByVertex.height);
            
            frameIndex=0;
            globalRotations.clear();
            frameStamp=timestamp+cameraSettings.td;
            
            imu.getIntraFrameRotation(rotations,grids.originGridVertices,frameSize,sizeByVertex,frameIndex,frameStamp);
            rotations[rotations.size()-1].copyTo(rotation);
            globalRotations.push_back(cv::Mat::eye(3,3,CV_64FC1));
            
            //create klt frame
            pyramidBuffers.initialize();
            computePyramid(trackingContext,im,*pyramidBuffers.ptrs[1]);
            
            pyramidBuffers.preloadThread=std::thread(computePyramid,
                                                     std::ref(trackingContext),
                                                     std::ref(*preloadImage),
                                                     std::ref(*pyramidBuffers.ptrs[2]));
            //char name[200];
            //printf(name,"/Users/ctang/edge.png");
            /*cv::Mat normx;
            cv::Mat normy;
            cv::Mat normxy;
            
            cv::multiply(pyramid1[1],pyramid1[1],normx);
            cv::multiply(pyramid1[2],pyramid1[2],normy);

            cv::sqrt(normx+normy,normxy);
            cv::imwrite("/Users/ctang/edge.png",normxy);getchar();*/
            /*cv::Mat gx,gy,gxx,gyy,gxy,sumGxx,sumGyy,sumGxy;
            cv::multiply(pyramid1[1],pyramid1[1],gxx);
            cv::multiply(pyramid1[2],pyramid1[2],gyy);
            cv::multiply(pyramid1[1],pyramid1[2],gxy);
            
            cv::integral(gxx,sumGxx,CV_32F);
            cv::integral(gyy,sumGyy,CV_32F);
            cv::integral(gxy,sumGxy,CV_32F);
            
            KLTSelectGoodFeatures(trackingContext,sumGxx,sumGyy,sumGxy,featureList);*/

            /*pyramid1Ptr=&pyramid1;
            pyramid2Ptr=&pyramid2;
            pyramid3Ptr=&pyramid3;
            pyramidTmpPtr=NULL;*/
            
            
            
            
            //create orb frame and keyframe
            cv::Mat ucharImage;
            //pyramid1[3*(trackingContext->nPyramidLevels-1)].convertTo(ucharImage,CV_8UC1);
            (*pyramidBuffers.ptrs[1])[3*(trackingContext->nPyramidLevels-1)].convertTo(ucharImage,CV_8UC1);
            
            Frame *framePtr=new Frame(ucharImage,timestamp,mpORBExtractor,mpVocabulary,mK,mDistCoef);
            KeyFrame *keyFramePtr=new KeyFrame(framePtr,mpKeyFrameDatabase,invK,mScaleFactor);
            keyFramePtr->frameId=frameId;
            keyFramePtr->outId=outIndex;
            
            
            for(int i=0;i<keyFramePtr->mvLocalMapPoints.size();i++){
                featureList->feature[i]->x=keyFramePtr->framePtr->mvKeys[i].pt.x*mScaleFactor;
                featureList->feature[i]->y=keyFramePtr->framePtr->mvKeys[i].pt.y*mScaleFactor;
                featureList->feature[i]->val=KLT_TRACKED;
            }
            featureList->nFeatures=keyFramePtr->mvLocalMapPoints.size();
            frontKeyFrame=keyFramePtr;
            preKeyFrame=NULL;
            
            /*for(int i=0;i<frontKeyFrame->mvLocalMapPoints.size();i++){
                
                cv::Mat dst;
                grids.rotateAndNormalizePoint(cv::Point2f(featureList->feature[i]->x,featureList->feature[i]->y),
                                              dst,rotations,cvInvK);
                
                frontKeyFrame->mvLocalMapPoints[i].norm(0)=dst.at<double>(0);
                frontKeyFrame->mvLocalMapPoints[i].norm(1)=dst.at<double>(1);
                frontKeyFrame->mvLocalMapPoints[i].norm(2)=dst.at<double>(2);
                frontKeyFrame->mvLocalMapPoints[i].vec=frontKeyFrame->mvLocalMapPoints[i].norm
                                                      /frontKeyFrame->mvLocalMapPoints[i].norm(2);
                
                cv::Vec3b color=colorImage->at<cv::Vec3b>(round(featureList->feature[i]->y/2),round(featureList->feature[i]->x/2));
                frontKeyFrame->mvLocalMapPoints[i].color[0]=color.val[0];
                frontKeyFrame->mvLocalMapPoints[i].color[1]=color.val[1];
                frontKeyFrame->mvLocalMapPoints[i].color[2]=color.val[2];
                frontKeyFrame->mvLocalMapPoints[i].tracked.push_back(cv::Point2f(featureList->feature[i]->x,
                                                                                 featureList->feature[i]->y));
            }*/
            
            tbb::parallel_for(tbb::blocked_range<size_t>(0,frontKeyFrame->mvLocalMapPoints.size()),NormalizeKeyFrameInvoker(frontKeyFrame,&grids,&featureList,&rotations,&cvInvK,colorImage));
            
            //lightFramePre=new Frame(ucharImage,timestamp,mpORBExtractorFrame,mpVocabulary,mK,mDistCoef);
        
        }else{
            
            
            /*
            cv::Mat ucharImage;
            (*pyramid2Ptr)[3*(trackingContext->nPyramidLevels-1)].convertTo(ucharImage,CV_8UC1);
            lightFrameCur=new Frame(ucharImage,timestamp,mpORBExtractorFrame,mpVocabulary,mK,mDistCoef);
            */
            
            
            frameStamp=timestamp+cameraSettings.td;
            imu.getInterFrameRotation(rotation,frameIndex,preFrameStamp,frameStamp);
            globalRotations.push_back(rotation*globalRotations[frameId-1]);
            
            /*
            std::vector<cv::Point2f> pts1,pts2;
            cv::Mat transform=mK*rotation*mKInv;
            orbMatcher->SearchByRotation(lightFramePre,lightFrameCur,transform,pts1,pts2,48);
            std::vector<float> intensity1(pts1.size()),intensity2(pts2.size());
            for (int p=0;p<pts1.size();p++) {
                intensity1[p]=(*pyramid1Ptr)[3*(trackingContext->nPyramidLevels-1)].at<float>((int)pts1[p].y,
                                                                                              (int)pts1[p].x);
                
                intensity2[p]=(*pyramid2Ptr)[3*(trackingContext->nPyramidLevels-1)].at<float>((int)pts2[p].y,
                                                                                              (int)pts2[p].x);
            }
            compensateLightingAndMotion(pts1,pts2,intensity1,intensity2);
            */
            
            /*for(int i=0;i<9;i++){
                std::cout<<transform.at<double>(i)<<std::endl;
            }*/
            
            
            imu.getIntraFrameRotation(rotations,grids.originGridVertices,
                                      frameSize,sizeByVertex,frameIndex,frameStamp);
            rotations[rotations.size()-1].copyTo(rotation);
            
#ifdef DEBUG
            
            /*cv::Mat debugImage;
            cv::cvtColor((*pyramid1Ptr)[3*(trackingContext->nPyramidLevels-1)],debugImage,CV_GRAY2BGR);
            
            for (int p=0;p<pts1.size();p++) {
                cv::circle(debugImage,pts1[p],2,CV_RGB(0,0,255));
                cv::line(debugImage,pts1[p],pts2[p],CV_RGB(255,0,0));
            }
            
            char savename[200];
            sprintf(savename,"/Users/chaos/Desktop/debug/match_%d.png",frameId);
            cv::imwrite(savename,debugImage);
            debugImage.release();*/
#endif
            pyramidBuffers.next();
            pyramid1Ptr=pyramidBuffers.ptrs[0];
            pyramid2Ptr=pyramidBuffers.ptrs[1];
            
            //computePyramid3(trackingContext,im,*pyramid2Ptr);
            //computePyramid3(trackingContext,im,*pyramid2Ptr);
            
            pyramidBuffers.preloadThread=std::thread(computePyramid,
                                                     std::ref(trackingContext),
                                                     std::ref(*preloadImage),
                                                     std::ref(*pyramidBuffers.ptrs[2]));
            
            
            //match
            KLTTrackFeatures(trackingContext,
                             *pyramid1Ptr,
                             *pyramid2Ptr,
                             featureList,invK);
            
            /*for (int i=0;i<featureList->nFeatures;i++) {
                if (featureList->feature[i]->val==KLT_TRACKED) {
                    
                    cv::Mat dst;
                    grids.rotateAndNormalizePoint(cv::Point2f(featureList->feature[i]->x,
                                                              featureList->feature[i]->y),
                                                  dst,rotations,cvInvK);
                    
                    featureList->feature[i]->norm(0)=dst.at<double>(0);
                    featureList->feature[i]->norm(1)=dst.at<double>(1);
                    featureList->feature[i]->norm(2)=dst.at<double>(2);
                    featureList->feature[i]->vec=featureList->feature[i]->norm
                                                /featureList->feature[i]->norm(2);
                    
                    frontKeyFrame->mvLocalMapPoints[i].tracked.push_back(cv::Point2f(featureList->feature[i]->x,
                                                                                     featureList->feature[i]->y));
                }
            }*/
            
            tbb::parallel_for(tbb::blocked_range<size_t>(0,featureList->nFeatures),
                              NormalizeFrameInvoker(frontKeyFrame,&grids,&featureList,&rotations,&cvInvK));
            
            
            KeyFrame* keyFrame=frontKeyFrame;
            int pVectorCount=0,fullTrackCount=0,inlierCount=0;
            std::vector<bool> isValid(keyFrame->mvLocalMapPoints.size());
            for (int i=0;i<keyFrame->mvLocalMapPoints.size();i++) {
                isValid[i]=keyFrame->mvLocalMapPoints[i].isFullTrack;
            }
                
            RelativeOutlierRejection outlierRejection;
            outlierRejection.K=K;
            for (int i1=0;i1<3;i1++) {
                for (int i2=0;i2<3;i2++) {
                    outlierRejection.rotation(i1,i2)=globalRotations[globalRotations.size()-1].at<double>(i1,i2);
                }
            }
            
            outlierRejection.rotation=outlierRejection.rotation*keyFrame->pose.rotation.transpose();
            outlierRejection.theshold=10.0;
            outlierRejection.prob=0.98;
            outlierRejection.medViewAngle=slamSettings.medViewAngle;
            outlierRejection.minViewAngle=slamSettings.minViewAngle;
            outlierRejection.isValid=&isValid;
            inlierCount=outlierRejection.relativeRansac(keyFrame,featureList);
            
            RelativePoseEstimation relativeEstimation;
            std::vector<Eigen::Vector3d*> pVectors;
            
            /*static Eigen::Matrix3d prevRotation;
            static Eigen::Vector3d prevTranslation;
            
            if (keyFrame->mvLocalFrames.size()==0) {
                outlierRejection.rotation=Eigen::Matrix3d::Identity();
                prevTranslation=Eigen::Vector3d::Zero();
            }else{
                outlierRejection.rotation=prevRotation;
            }*/
            
            if(inlierCount>0){
                relativeEstimation.isValid=&isValid;
                
                relativeEstimation.rotation=outlierRejection.rotation;
                //relativeEstimation.translation=prevTranslation;
                
                relativeEstimation.rotationIsKnown=true;
                relativeEstimation.medViewAngle=slamSettings.medViewAngle;
                relativeEstimation.minViewAngle=slamSettings.minViewAngle;
                pVectorCount=relativeEstimation.estimateRelativePose(keyFrame,featureList,pVectors);
            }
            //prevRotation=relativeEstimation.rotation;
            //prevTranslation=relativeEstimation.translation;
            
                
            bool requestNewKeyFrame=false;
            bool isLastKeyFrame=false;
            
            if (pVectorCount>0) {
                
                fullTrackCount=keyFrame->appendRelativeEstimation(frameId,NULL,
                                                                  relativeEstimation.rotation,
                                                                  relativeEstimation.translation,
                                                                  featureList,pVectors);
                
                requestNewKeyFrame=fullTrackCount<slamSettings.requireNewKeyFrameCount
                                   &&keyFrame->mvLocalFrames.size()>15;
                //requestNewKeyFrame=(requestNewKeyFrame||keyFrame->mvLocalFrames.size()>40);
                isLastKeyFrame=true;
            }
            
            
            if(requestNewKeyFrame){
                
                printf("new %d old %d\n",frameId,frontKeyFrame->frameId);
                char name[200];
                
                /*LocalFactorization localFactorization;
                localFactorization.process(keyFrame);
                //localFactorization.iterativeRefine(keyFrame);
                
#ifdef DEBUG
                //sprintf(name,"/Users/chaos/Desktop/debug/init_%d_%d.ply",frontKeyFrame->frameId,frameId);
                //keyFrame->savePly(name);
#endif
                
                LocalBundleAdjustment localBundleAdjustment;
                localBundleAdjustment.projErrorThres=0.005;
                localBundleAdjustment.viewAngleThres=slamSettings.minViewAngle;
                //printf("ba\n");
                
                localBundleAdjustment.bundleAdjust(keyFrame);
                
                
                //sprintf(name,"/Users/chaos/Desktop/debug/ba0_%d_%d.ply",frontKeyFrame->frameId,frameId);
                //keyFrame->savePly(name);
                
                localBundleAdjustment.triangulate2(keyFrame);
                
                //sprintf(name,"/Users/chaos/Desktop/debug/tra_%d_%d.ply",frontKeyFrame->frameId,frameId);
                //keyFrame->savePly(name);

                
                //localBundleAdjustment.bundleAdjust(keyFrame);
                localBundleAdjustment.refinePoints(keyFrame);
                
                //keyFrame->visualize();
                
                //localBundleAdjustment.estimateCloseFrames(keyFrame);
                //getchar();
#ifdef DEBUG
                //sprintf(name,"/Users/chaos/Desktop/debug/ba1_%d_%d.ply",frontKeyFrame->frameId,frameId);
                //keyFrame->savePly(name);
                //getchar();
#endif

                
                if(preKeyFrame!=NULL){
                    keyFrameConnector.mpORBVocabulary=mpVocabulary;
                    keyFrameConnector.keyFrameDatabase=mpKeyFrameDatabase;
                    keyFrameConnector.connectionThreshold=80;
                    keyFrameConnector.connectKeyFrame(preKeyFrame,keyFrame);
                }*/
                keyFrame->prevKeyFramePtr=preKeyFrame;
                keyFrame->baThread=std::thread(processKeyFrame,std::ref(*this),std::ref(*keyFrame));
                //processKeyFrame(*this,*keyFrame);
                
                
                //mpKeyFrameDatabase->add(keyFrame);
                //globalReconstruction.addNewKeyFrame(keyFrame);
                
                
                cv::Mat ucharImage;
                (*pyramid2Ptr)[3*(trackingContext->nPyramidLevels-1)].convertTo(ucharImage,CV_8UC1);
                Frame *framePtr=new Frame(ucharImage,timestamp,mpORBExtractor,mpVocabulary,mK,mDistCoef);
                KeyFrame *keyFramePtr=new KeyFrame(framePtr,mpKeyFrameDatabase,invK,mScaleFactor);
                keyFramePtr->frameId=frameId;
                keyFramePtr->outId=outIndex;
                
                //set rotation and derolling shutter
                for (int i1=0;i1<3;i1++) {
                    for (int i2=0;i2<3;i2++) {
                        keyFramePtr->pose.rotation(i1,i2)=globalRotations[globalRotations.size()-1].at<double>(i1,i2);
                    }
                }
                for(int i=0;i<keyFramePtr->mvLocalMapPoints.size();i++){
                    featureList->feature[i]->x=keyFramePtr->framePtr->mvKeys[i].pt.x*mScaleFactor;
                    featureList->feature[i]->y=keyFramePtr->framePtr->mvKeys[i].pt.y*mScaleFactor;
                    featureList->feature[i]->val=KLT_TRACKED;
                }
                featureList->nFeatures=keyFramePtr->mvLocalMapPoints.size();
                
                /*for(int i=0;i<keyFramePtr->mvLocalMapPoints.size();i++){
                    
                    cv::Mat dst;
                    grids.rotateAndNormalizePoint(cv::Point2f(featureList->feature[i]->x,
                                                              featureList->feature[i]->y),
                                                  dst,rotations,cvInvK);
                    
                    keyFramePtr->mvLocalMapPoints[i].norm(0)=dst.at<double>(0);
                    keyFramePtr->mvLocalMapPoints[i].norm(1)=dst.at<double>(1);
                    keyFramePtr->mvLocalMapPoints[i].norm(2)=dst.at<double>(2);
                    keyFramePtr->mvLocalMapPoints[i].vec=keyFramePtr->mvLocalMapPoints[i].norm
                                                        /keyFramePtr->mvLocalMapPoints[i].norm(2);
                    
                    
                    cv::Vec3b color=colorImage->at<cv::Vec3b>(round(featureList->feature[i]->y/2),
                                                              round(featureList->feature[i]->x/2));
                    keyFramePtr->mvLocalMapPoints[i].color[0]=color.val[0];
                    keyFramePtr->mvLocalMapPoints[i].color[1]=color.val[1];
                    keyFramePtr->mvLocalMapPoints[i].color[2]=color.val[2];
                    
                    keyFramePtr->mvLocalMapPoints[i].tracked.push_back(cv::Point2f(featureList->feature[i]->x,
                                                                                   featureList->feature[i]->y));
                }*/
                
                
                 tbb::parallel_for(tbb::blocked_range<size_t>(0,keyFramePtr->mvLocalMapPoints.size()),NormalizeKeyFrameInvoker(keyFramePtr,&grids,&featureList,&rotations,&cvInvK,colorImage));
                
                preKeyFrame=frontKeyFrame;
                frontKeyFrame=keyFramePtr;
                frontKeyFrame->prevKeyFramePtr=preKeyFrame;
            }
            printf("frame %d end\n",frameId);
            pyramidTmpPtr=pyramid1Ptr;
            pyramid1Ptr=pyramid2Ptr;
            pyramid2Ptr=pyramidTmpPtr;
            
            //delete lightFramePre;
            //lightFramePre=lightFrameCur;
        }
        preFrameStamp=frameStamp;
        frameId++;
        return pose;
    }
    
    void System::finish(){
        globalReconstruction.path=path;
        pyramidBuffers.preloadThread.join();
        if(frontKeyFrame->mvLocalFrames.size()>0){
            
            LocalFactorization localFactorization;
            localFactorization.process(frontKeyFrame);
        
            LocalBundleAdjustment localBundleAdjustment;
            localBundleAdjustment.projErrorThres=0.005;
            localBundleAdjustment.viewAngleThres=slamSettings.minViewAngle;
            localBundleAdjustment.bundleAdjust(frontKeyFrame);
        
            localBundleAdjustment.triangulate(frontKeyFrame);
            localBundleAdjustment.bundleAdjust(frontKeyFrame);
            
            char name[200];
            sprintf(name,"/Users/chaos/Desktop/syndata/data_%d.txt",frontKeyFrame->frameId);
            frontKeyFrame->saveData2(name);
            sprintf(name,"/Users/chaos/Desktop/syndata/data_%d.ply",frontKeyFrame->frameId);
            frontKeyFrame->savePly(name);
            
            if(preKeyFrame->baThread.joinable()){
                preKeyFrame->baThread.join();
            }
            
            keyFrameConnector.mpORBVocabulary=mpVocabulary;
            keyFrameConnector.keyFrameDatabase=mpKeyFrameDatabase;
            keyFrameConnector.connectionThreshold=80;
            keyFrameConnector.connectKeyFrame(preKeyFrame,frontKeyFrame);
            
            mpKeyFrameDatabase->add(frontKeyFrame);
            globalReconstruction.addNewKeyFrame(frontKeyFrame);
        }        
        //globalReconstruction.estimateSIM3();
        //globalReconstruction.savePly();
        //globalReconstruction.savePly();
        
        
        
        globalReconstruction.scaleThreshold=20;
        globalReconstruction.estimateScale();
        std::vector<int> index;
        globalReconstruction.estimateRotation(index);
        printf("rotation estiamted\n");
        globalReconstruction.estimateTranslation(index);
        globalReconstruction.globalRefine();
        globalReconstruction.savePly();
        
        
        //globalReconstruction.frameStart=frameStart;
        globalReconstruction.visualize();
        //globalReconstruction.topview();
        
        //globalReconstruction.savePly();
        
        //globalReconstruction.topview()
        //globalReconstruction.savePly();
        //globalReconstruction.visualize();
        //globalReconstruction.savePly();
        //globalReconstruction.globalPBA();
        //globalReconstruction.estimateSIM3();
    }
}
