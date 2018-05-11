//
//  Drawer.h
//  GSLAM
//
//  Created by ctang on 11/19/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#ifndef DRAWER_H
#define DRAWER_H

#include "pangolin/pangolin.h"
#include "ceres/rotation.h"
#include <vector>
#include <utility>

namespace GSLAM
{
    class KeyFrame;
    class Drawer{
    public:
        float mCameraSize;
        float mCameraLineWidth;
        float mGraphLineWidth;
        float mPointSize;
        
        int   mFrameIndex;
        int   mKeyFrameIndex;
        std::vector<KeyFrame*> keyFrames;
        
        void drawCurrentCamera(pangolin::OpenGlMatrix &Twc);
        void drawCurrentTrajectory(pangolin::OpenGlMatrix &Twc);

        void getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &Twc);
        void drawKeyFrames();
        void drawPoints();
        std::vector<pangolin::OpenGlMatrix> preTwc;
    };
}

#include "KeyFrame.h"
namespace GSLAM{
    
    void Drawer::drawCurrentTrajectory(pangolin::OpenGlMatrix &Twc) {
        
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;
        
        glPushMatrix();
        glMultMatrixd(Twc.m);
        
        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f/255.0,114.0f/255.0,189.0f/255.0);
        
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);
        
        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);
        
        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);
        
        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);
        
        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();
        
        glPopMatrix();
        
        preTwc.push_back(Twc);
        
        glPointSize(4);
        glBegin(GL_POINTS);
        for(int i=0;i<preTwc.size();i++){
            glVertex3d(preTwc[i].m[12],preTwc[i].m[13],preTwc[i].m[14]);
        }
        glEnd();
    }
    
    void Drawer::drawCurrentCamera(pangolin::OpenGlMatrix &Twc) {
        
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;
        
        glPushMatrix();
        glMultMatrixd(Twc.m);
        
        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f/255.0,114.0f/255.0,189.0f/255.0);
        
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);
        
        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);
        
        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);
        
        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);
        
        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();
        glPopMatrix();
        
        glPointSize(4);
        glBegin(GL_POINTS);
        for(int i=0;i<preTwc.size();i++){
            glVertex3d(preTwc[i].m[12],preTwc[i].m[13],preTwc[i].m[14]);
        }
        glEnd();
        preTwc.push_back(Twc);
    }
    void Drawer::getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M){
        
        if(mFrameIndex>=keyFrames[mKeyFrameIndex]->mvLocalFrames.back().frameId){
            preTwc.clear();
            mKeyFrameIndex++;
        }
        //printf("%d %d\n",mFrameIndex,keyFrames[mKeyFrameIndex]->frameId);
        
        Transform pose;
        Transform pose1;
        Transform pose2;
        
        if(mKeyFrameIndex<1){
            
            pose1=keyFrames[mKeyFrameIndex]->pose;
            pose2=keyFrames[mKeyFrameIndex]->pose;
            
        }else{

            pose1.rotation=keyFrames[mKeyFrameIndex-1]->mvLocalFrames.back().pose.rotation
                          *keyFrames[mKeyFrameIndex-1]->pose.rotation;
            
            Eigen::Vector3d translation;
            translation =keyFrames[mKeyFrameIndex-1]->mvLocalFrames.back().pose.translation
                        /keyFrames[mKeyFrameIndex-1]->scale;
            
            translation =keyFrames[mKeyFrameIndex-1]->pose.rotation.transpose()*translation;
            pose1.translation=keyFrames[mKeyFrameIndex-1]->pose.translation+translation;
            
            pose2=keyFrames[mKeyFrameIndex]->pose;
        }
        
        if (mFrameIndex==keyFrames[mKeyFrameIndex]->frameId) {
            
            pose=pose1;
            
        }else if (mFrameIndex<keyFrames[mKeyFrameIndex]->mvLocalFrames[0].frameId) {
            
            double distance1=mFrameIndex-keyFrames[mKeyFrameIndex]->frameId;
            double distance2=keyFrames[mKeyFrameIndex]->mvLocalFrames[0].frameId-mFrameIndex;
            
            Eigen::Vector3d angle1,angle2,angle;
            ceres::RotationMatrixToAngleAxis(pose1.rotation.data(),angle1.data());
            ceres::RotationMatrixToAngleAxis(pose2.rotation.data(),angle2.data());
            angle=(distance2*angle1+distance1*angle2)/(distance1+distance2);
            
            ceres::AngleAxisToRotationMatrix(angle.data(),pose.rotation.data());
            pose.translation=(distance2*pose1.translation+distance1*pose2.translation)/(distance1+distance2);
            
            
            pose1=keyFrames[mKeyFrameIndex]->mvLocalFrames[0].pose;
            ceres::RotationMatrixToAngleAxis(pose1.rotation.data(),angle1.data());
            angle1*=(distance1/(distance1+distance2));
            ceres::AngleAxisToRotationMatrix(angle1.data(),pose1.rotation.data());
            
            pose1.translation=keyFrames[mKeyFrameIndex]->mvLocalFrames[0].pose.translation/keyFrames[mKeyFrameIndex]->scale;
            pose1.translation*=(distance1/(distance1+distance2));
            
            pose.rotation=pose1.rotation*pose.rotation;
            pose.translation+=pose.rotation.transpose()*pose1.translation;
            
        }else{
            
            int localIndex=mFrameIndex-keyFrames[mKeyFrameIndex]->mvLocalFrames[0].frameId;
            pose.rotation=keyFrames[mKeyFrameIndex]->mvLocalFrames[localIndex].pose.rotation
                         *keyFrames[mKeyFrameIndex]->pose.rotation;
            
            Eigen::Vector3d translation;
            translation =keyFrames[mKeyFrameIndex]->mvLocalFrames[localIndex].pose.translation
                        /keyFrames[mKeyFrameIndex]->scale;
            translation =keyFrames[mKeyFrameIndex]->pose.rotation.transpose()*translation;
            pose.translation=pose2.translation+translation;
        }
        
        pose.rotation.transposeInPlace();
        
        M.m[0] = pose.rotation(0,0);
        M.m[1] = pose.rotation(1,0);
        M.m[2] = pose.rotation(2,0);
        M.m[3] = 0.0;
        
        M.m[4] = pose.rotation(0,1);
        M.m[5] = pose.rotation(1,1);
        M.m[6] = pose.rotation(2,1);
        M.m[7]  = 0.0;
        
        M.m[8]  = pose.rotation(0,2);
        M.m[9]  = pose.rotation(1,2);
        M.m[10] = pose.rotation(2,2);
        M.m[11] = 0.0;
        
        M.m[12] = pose.translation(0);
        M.m[13] = pose.translation(1);
        M.m[14] = pose.translation(2);
        M.m[15] = 1.0;
        
        mFrameIndex++;
    }
    
    void Drawer::drawPoints(){
        glPointSize(4);
        glBegin(GL_POINTS);
        for(int k=0;k<mKeyFrameIndex;k++){
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if(keyFrames[k]->mvLocalMapPoints[i].isEstimated){
                    Eigen::Vector3d point3D=keyFrames[k]->mvLocalMapPoints[i].getPosition();
                    point3D/=keyFrames[k]->scale;
                    point3D=keyFrames[k]->pose.rotation.transpose()*point3D+keyFrames[k]->pose.translation;
                    uchar* color=keyFrames[k]->mvLocalMapPoints[i].color;
                    glColor3f(color[2]*(1.0/255.0),color[1]*(1.0/255.0),color[0]*(1.0/255.0));
                    glVertex3d(point3D(0),point3D(1),point3D(2));
                }
            }
        }
        glEnd();
    }
    
    void Drawer::drawKeyFrames(){
        
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;
        
        std::vector<Eigen::Vector3d> centers(0);
        for(size_t i=0; i<=mKeyFrameIndex; i++){
            
            KeyFrame* pKF = keyFrames[i];
            Transform pose;
            
            if(i==0){
                pose=pKF->pose;
            }else{
                pose.rotation=keyFrames[i-1]->mvLocalFrames.back().pose.rotation
                             *keyFrames[i-1]->pose.rotation;
                
                Eigen::Vector3d translation;
                translation =keyFrames[i-1]->mvLocalFrames.back().pose.translation/keyFrames[i-1]->scale;
                
                translation =keyFrames[i-1]->pose.rotation.transpose()*translation;
                pose.translation=keyFrames[i-1]->pose.translation+translation;
            }
            pose.rotation.transposeInPlace();
            centers.push_back(pose.translation);
            
            pangolin::OpenGlMatrix M;
            M.m[0] = pose.rotation(0,0);
            M.m[1] = pose.rotation(1,0);
            M.m[2] = pose.rotation(2,0);
            M.m[3] = 0.0;
            
            M.m[4] = pose.rotation(0,1);
            M.m[5] = pose.rotation(1,1);
            M.m[6] = pose.rotation(2,1);
            M.m[7]  = 0.0;
            
            M.m[8]  = pose.rotation(0,2);
            M.m[9]  = pose.rotation(1,2);
            M.m[10] = pose.rotation(2,2);
            M.m[11] = 0.0;
            
            M.m[12] = pose.translation(0);
            M.m[13] = pose.translation(1);
            M.m[14] = pose.translation(2);
            M.m[15] = 1.0;
            
            glPushMatrix();
            glMultMatrixd(M.m);
            
            glLineWidth(mCameraLineWidth*1.5);
            glColor3f(0.0f/255.0,114.0f/255.0,189.0f/255.0);
            
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);
            
            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);
            
            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);
            
            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);
            
            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();
            glPopMatrix();
        }
        
        glLineWidth(mGraphLineWidth);
        
        glBegin(GL_LINES);
        //printf("show %d keyframes\n",centers.size());
        std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d> > lines;
        for(size_t k=0;k<=mKeyFrameIndex;k++)
        {
            // Covisibility Graph
            Eigen::Vector3d center=centers[k];//=keyFrames[k]->pose.translation;
            for(map<KeyFrame*,Transform>::iterator mit=keyFrames[k]->mConnectedKeyFramePoses.begin(),
                mend=keyFrames[k]->mConnectedKeyFramePoses.end();
                mit!=mend;
                mit++){
                
                if(mit->first->mnId<keyFrames[k]->mnId){
                    if(mit->first->nextKeyFramePtr==keyFrames[k]&&mit->first==keyFrames[k]->prevKeyFramePtr){
                        glColor3f(0.0f/255.0,114.0f/255.0,189.0f/255.0);
                        Eigen::Vector3d center2=centers[mit->first->mnId];//mit->first->pose.translation;
                        glVertex3d(center(0),center(1),center(2));
                        glVertex3d(center2(0),center2(1),center2(2));
                        
                    }else if(mit->second.scale==-1.0){
                        Eigen::Vector3d center2=centers[mit->first->mnId];
                        if(mit->first->frameId<70||keyFrames[k]->frameId<70){
                            lines.push_back(std::make_pair(center,center2));
                            continue;
                        }
                        glColor3f(162.0f/255.0,20.0f/255.0,47.0f/255.0);
                        glVertex3d(center(0),center(1),center(2));
                        glVertex3d(center2(0),center2(1),center2(2));

                    }else{
                        glColor3f(237.0f/255.0,177.0f/255.0,32.0f/255.0);
                        Eigen::Vector3d center2=centers[mit->first->mnId];//mit->first->pose.translation;
                        glVertex3d(center(0),center(1),center(2));
                        glVertex3d(center2(0),center2(1),center2(2));
                    }
                }
            }
        }
        glEnd();
        
        glLineWidth(mGraphLineWidth);
        glLineStipple(8, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        
        for(int i=0;i<lines.size();i++){
            Eigen::Vector3d center1=lines[i].first;
            Eigen::Vector3d center2=lines[i].second;
            glColor3f(162.0f/255.0,20.0f/255.0,47.0f/255.0);
            glVertex3d(center1(0),center1(1),center1(2));
            glVertex3d(center2(0),center2(1),center2(2));
        }
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }
}

#endif