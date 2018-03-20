//
//  KeyFrame.cpp
//  GSLAM
//
//  Created by ctang on 9/4/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "KeyFrame.h"
#include "Estimation.h"
#include "Geometry.h"
#include "MapPoint.h"
//#include "Drawer.h"
//#include <unistd.h>
//#include "opencv2/viz.hpp"

namespace GSLAM{
    
    long unsigned int KeyFrame::nNextId=0;

    KeyFrame::KeyFrame(Frame *F,KeyFrameDatabase *pKFDB,const Eigen::Matrix3d &invK,double _mScaleFactor):
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0),
    mBowVec(F->mBowVec), mFeatVec(F->mFeatVec),mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F->mpORBvocabulary),mScaleFactor(_mScaleFactor){
        
        mnId=nNextId++;
        framePtr=F;
        F->keyFramePtr=this;
        
        //initialize for geometry
        mvRelativeEstimated.resize(framePtr->mvKeys.size());
        std::fill(mvRelativeEstimated.begin(),mvRelativeEstimated.end(),true);
        mvLocalFrames.clear();
        mvLocalMapPoints.resize(framePtr->mvKeys.size());
        
        for (int i=0;i<mvLocalMapPoints.size();i++) {
            
            Eigen::Vector3d kpt((double)framePtr->mvKeys[i].pt.x*mScaleFactor,
                                (double)framePtr->mvKeys[i].pt.y*mScaleFactor,1.0);
            kpt=invK*kpt;
            
            //mvLocalMapPoints[i].pt=framePtr->mvKeys[i].pt*mScaleFactor;
            mvLocalMapPoints[i].norm=kpt;
            mvLocalMapPoints[i].norm.normalize();

            
            mvLocalMapPoints[i].vec=kpt;
            mvLocalMapPoints[i].vec/=mvLocalMapPoints[i].vec(2);
            
            mvLocalMapPoints[i].vecs.clear();
            mvLocalMapPoints[i].pVectors.clear();
            mvLocalMapPoints[i].isFullTrack=true;
            mvLocalMapPoints[i].isEstimated=false;
            mvLocalMapPoints[i].measurementCount=0;
            
            const int level   = framePtr->mvKeysUn[i].octave;
            const int nLevels = framePtr->mnScaleLevels;
            
            mvLocalMapPoints[i].maxLevelScaleFactor=framePtr->mvScaleFactors[level];
            mvLocalMapPoints[i].minLevelScaleFactor=framePtr->mvScaleFactors[nLevels-1];
#ifdef DEBUG
            //std::cout<<invK<<std::endl;
            //std::cout<<kpt<<std::endl;
            //std::cout<<mvLocalMapPoints[i].norm<<std::endl;
#endif DEBUG
        }
        
        scale=1.0;
        logScale=0.0;
        minScore=1.0;
        
        
        pose.rotation=Eigen::Matrix3d::Identity();
        pose.translation=Eigen::Vector3d::Zero();
        
        isGlobalFixed=false;
        nextKeyFramePtr=NULL;
        prevKeyFramePtr=NULL;
    }
    
    KeyFrame::KeyFrame(Frame *F,KeyFrameDatabase *pKFDB,const Eigen::Matrix3d &invK,KLT_FeatureList featureList):
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0),
    mBowVec(F->mBowVec), mFeatVec(F->mFeatVec),mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F->mpORBvocabulary){
        
        mnId=nNextId++;
        framePtr=F;
        F->keyFramePtr=this;
        
        //initialize for geometry
        mvRelativeEstimated.resize(featureList->nFeatures);
        std::fill(mvRelativeEstimated.begin(),mvRelativeEstimated.end(),true);
        mvLocalFrames.clear();
        mvLocalMapPoints.resize(featureList->nFeatures);
        
        for (int i=0;i<mvLocalMapPoints.size();i++) {
            
            Eigen::Vector3d kpt(featureList->feature[i]->x,featureList->feature[i]->y,1.0);
            kpt=invK*kpt;
            
            mvLocalMapPoints[i].norm=kpt;
            mvLocalMapPoints[i].norm.normalize();
            
            
            mvLocalMapPoints[i].vec=kpt;
            mvLocalMapPoints[i].vec/=mvLocalMapPoints[i].vec(2);
            
            //mvLocalMapPoints[i].pt.x=featureList->feature[i]->x;
            //mvLocalMapPoints[i].pt.y=featureList->feature[i]->y;
            
            
            mvLocalMapPoints[i].vecs.clear();
            mvLocalMapPoints[i].pVectors.clear();
            mvLocalMapPoints[i].isFullTrack=true;
            mvLocalMapPoints[i].isEstimated=false;
            mvLocalMapPoints[i].measurementCount=0;
            
#ifdef DEBUG
            //std::cout<<invK<<std::endl;
            //std::cout<<kpt<<std::endl;
            //std::cout<<mvLocalMapPoints[i].norm<<std::endl;
#endif DEBUG
            
        }
    }
    
    
    set<KeyFrame*> KeyFrame::GetConnectedKeyFrames(){
        unique_lock<mutex> lock(mMutexConnections);
        set<KeyFrame*> s;
        for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
            s.insert(mit->first);
        return s;
    }
    
    vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N){
        
        unique_lock<mutex> lock(mMutexConnections);
        
        if((int)mvpOrderedConnectedKeyFrames.size()<N)
            return mvpOrderedConnectedKeyFrames;
        else
            return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
        
    }
    
    

    
    int   KeyFrame::appendRelativeEstimation(const int frameId,KeyFrame *keyFrame,
                                             const Eigen::Matrix3d &rotation,const Eigen::Vector3d &translation,
                                             const KLT_FeatureList featureList,const vector<Eigen::Vector3d*> &pVectors){
        
        LocalFrame frame;
        frame.frameId=frameId;
        frame.measurementCount=0;
        frame.pose.rotation=rotation;
        frame.pose.translation=translation;
        
        int fullTrackCount=0;
        for (int i=0;i<mvLocalMapPoints.size();i++) {
            
            mvLocalMapPoints[i].pVectors.push_back(pVectors[i]);
            mvLocalMapPoints[i].isFullTrack&=(pVectors[i]!=NULL);
            fullTrackCount+=mvLocalMapPoints[i].isFullTrack;
            
            if (featureList->feature[i]->val==KLT_TRACKED) {
                Eigen::Vector3d* vecPtr=new Eigen::Vector3d(featureList->feature[i]->vec);
                mvLocalMapPoints[i].vecs.push_back(vecPtr);
                mvLocalMapPoints[i].measurementCount++;
                frame.measurementCount++;
            }else{
                mvLocalMapPoints[i].vecs.push_back(static_cast<Eigen::Vector3d*>(NULL));
            }
        }
        mvLocalFrames.push_back(frame);
        return fullTrackCount;
    }
    
    void KeyFrame::UpdateBestCovisibles(){
        vector<pair<int,KeyFrame*> > vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());
        for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end();
            mit!=mend;
            mit++){
            vPairs.push_back(make_pair(mit->second,mit->first));
        }
        
        sort(vPairs.begin(),vPairs.end());
        list<KeyFrame*> lKFs;
        list<int> lWs;
        for(size_t i=0, iend=vPairs.size(); i<iend;i++){
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
    }
    
    int KeyFrame::appendKeyFrame(KeyFrame *keyFrame,Transform transform,vector<int>& matches){
        
        int nInlierMatch=0;
        for (int i=0;i<matches.size();i++) {
            nInlierMatch+=(matches[i]>=0);
        }
        assert(!mConnectedKeyFrameWeights.count(keyFrame));
        printf("append %d to %d %d\n",this->frameId,keyFrame->frameId,nInlierMatch);
        if (nInlierMatch>=20) {
            if(!mConnectedKeyFrameWeights.count(keyFrame)){
                mConnectedKeyFrameWeights[keyFrame]=nInlierMatch;
                mConnectedKeyFramePoses[keyFrame]=transform;
                mConnectedKeyFrameMatches[keyFrame].resize(matches.size());
                std::copy(matches.begin(),matches.end(),mConnectedKeyFrameMatches[keyFrame].begin());
            }
            else if(mConnectedKeyFrameWeights[keyFrame]!=nInlierMatch){
                mConnectedKeyFrameWeights[keyFrame]=nInlierMatch;
                mConnectedKeyFramePoses[keyFrame]=transform;
                std::copy(matches.begin(),matches.end(),mConnectedKeyFrameMatches[keyFrame].begin());
            }
            else{
                return 0;
            }
            UpdateBestCovisibles();
        }else{
            assert(0);
        }
        //printf("%d append %d %d\n",frameId,keyFrame->frameId,nInlierMatch);
        //finally append frame
        
        /*for (int i=0;i<matches.size();i++) {
            if (matches[i]>=0) {
                printf("keyframe %d %d estimated matched keyframe %d %d estimated\n",
                       frameId,i,mvLocalMapPoints[i].isEstimated,
                       keyFrame->frameId,matches[i],keyFrame->mvLocalMapPoints[matches[i]].isEstimated);
            }
        }*/
        
        return  nInlierMatch;
    }
    
    
    void KeyFrame::savePly(const char* savename){
        
        int num_point=0;
        for(int i=0;i<mvLocalMapPoints.size();i++){
            if(mvLocalMapPoints[i].isEstimated){
                num_point++;
            }
        }
        
        int num_frame=mvLocalFrames.size()+mvpOrderedConnectedKeyFrames.size();
        
        std::ofstream of(savename);
        of << "ply"
        << '\n' << "format ascii 1.0"
        << '\n' << "element vertex " <<num_frame+num_point
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        << '\n' << "end_header" << std::endl;
        
        for(int i=0;i<mvLocalMapPoints.size();i++){
            if(mvLocalMapPoints[i].isEstimated){
                uchar *color=mvLocalMapPoints[i].color;
                Eigen::Vector3d position=mvLocalMapPoints[i].getPosition();
                of<<position(0)<<' '<<position(1)<<' '<<position(2)<<' '<<(int)color[2]<<' '<<(int)color[1]<<' '<<(int)color[0]<<std::endl;
            }
        }
        
        char * dotpos=strchr(savename,'.');
        dotpos[1]='t';
        dotpos[2]='x';
        dotpos[3]='t';
        
        std::ofstream of1(savename);
        if (!mvLocalFrames.empty()) {
            for(int f=0;f<mvLocalFrames.size();f++){
                of<<mvLocalFrames[f].pose.translation(0)<<' '<<mvLocalFrames[f].pose.translation(1)<<' '<<mvLocalFrames[f].pose.translation(2)<<" 255 255 255"<<endl;
                of1<<mvLocalFrames[f].pose.rotation<<endl;
                of1<<mvLocalFrames[f].pose.translation.transpose()<<endl;
            }
        }
        of1.close();
        
        
        std::ofstream record(savename);
        record<<num_frame<<std::endl;
        for (int i=0;i<num_frame;i++) {
            record<<mvLocalFrames[i].pose.rotation<<std::endl;
            record<<mvLocalFrames[i].pose.translation.transpose()<<std::endl;
        }
        
        for (int i=0;i<mvLocalMapPoints.size();i++) {
            if(mvLocalMapPoints[i].isEstimated){
                Eigen::Vector3d position=mvLocalMapPoints[i].getPosition();
                record<<mvLocalMapPoints[i].norm.transpose()<<std::endl;
                record<<position.transpose()<<std::endl;
            }
        }
        record.close();
        
        
        
        for(map<KeyFrame*,Transform>::iterator mit=mConnectedKeyFramePoses.begin(),
            mend=mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            
            Transform pose=mit->second;
            if (pose.scale==-1.0) {
                of<<pose.translation(0)<<' '<<pose.translation(1)<<' '<<pose.translation(2)<<" 0 255 0"<<endl;
            }else{
                of<<pose.translation(0)<<' '<<pose.translation(1)<<' '<<pose.translation(2)<<" 255 0 0"<<endl;
            }
            
        }
        of<<"0 0 0 255 0 0"<<endl;
        of.close();
    }
    
    void KeyFrame::saveData2(const char* savename){
        //int num_frame=mvLocalFrames.size();

    }
    
    void KeyFrame::visualize(){
        
        /*pangolin::CreateWindowAndBind("GSLAM: Map Viewer",1024,768);
        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);
        
        // Issue specific OpenGl we might need
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        //pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
        
        float mViewpointX=0;
        float mViewpointY=-0.7;
        float mViewpointZ=-1.8;
        float mViewpointF=500;
        
        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                                          pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
        
        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
        
        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();
        
        bool bFollow = true;
        bool bLocalizationMode = false;
        float mT = 1e3/20.0;
        Drawer drawer;
        drawer.mCameraSize=0.08;
        drawer.mCameraLineWidth=3;
        drawer.mFrameIndex=-1;
        drawer.mKeyFrameIndex=0;
        //drawer.keyFrames=keyFrames;
        int frame=0;
        while(1){
            
            //drawer.getCurrentOpenGLCameraMatrix(Twc);
            

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            s_cam.Follow(Twc);
            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);
            
            pose=mvLocalFrames[frame].pose;
            pose.rotation.transposeInPlace();
            
            Twc.m[0] = pose.rotation(0,0);
            Twc.m[1] = pose.rotation(1,0);
            Twc.m[2] = pose.rotation(2,0);
            Twc.m[3] = 0.0;
            
            Twc.m[4] = pose.rotation(0,1);
            Twc.m[5] = pose.rotation(1,1);
            Twc.m[6] = pose.rotation(2,1);
            Twc.m[7]  = 0.0;
            
            Twc.m[8]  = pose.rotation(0,2);
            Twc.m[9]  = pose.rotation(1,2);
            Twc.m[10] = pose.rotation(2,2);
            Twc.m[11] = 0.0;
            
            Twc.m[12] = pose.translation(0);
            Twc.m[13] = pose.translation(1);
            Twc.m[14] = pose.translation(2);
            Twc.m[15] = 1.0;
            

            
            for (int i=0;i<frame;i++) {
                
                pose=mvLocalFrames[i].pose;
                pose.rotation.transposeInPlace();
                
                Twc.m[0] = pose.rotation(0,0);
                Twc.m[1] = pose.rotation(1,0);
                Twc.m[2] = pose.rotation(2,0);
                Twc.m[3] = 0.0;
                
                Twc.m[4] = pose.rotation(0,1);
                Twc.m[5] = pose.rotation(1,1);
                Twc.m[6] = pose.rotation(2,1);
                Twc.m[7]  = 0.0;
                
                Twc.m[8]  = pose.rotation(0,2);
                Twc.m[9]  = pose.rotation(1,2);
                Twc.m[10] = pose.rotation(2,2);
                Twc.m[11] = 0.0;
                
                Twc.m[12] = pose.translation(0);
                Twc.m[13] = pose.translation(1);
                Twc.m[14] = pose.translation(2);
                Twc.m[15] = 1.0;
                drawer.drawCurrentCamera(Twc);
            }
            
            
            glPointSize(4);
            glBegin(GL_POINTS);
            for(size_t i=0;i<mvLocalMapPoints.size();i++){
                Eigen::Vector3d point3D=mvLocalMapPoints[i].getPosition();
                uchar *color=mvLocalMapPoints[i].color;
                
                glColor3f(color[2]/255.0,color[1]/255.0,color[0]/255.0);
                glVertex3f(point3D(0),point3D(1),point3D(2));
            }
            glEnd();
            
            if (frame<mvLocalFrames.size()-1) {
                frame++;
            }
            char name[200];
            sprintf(name,"//Users/ctang/Desktop/debug/test%d.png",frame);
            pangolin::SaveWindowOnRender(name);
            pangolin::FinishFrame();
            //sleep(30);
            //cv::waitKey(100);
        }*/
    }
}
