
//
//  GlobalReconstruction.cpp
//  GSLAM
//
//  Created by ctang on 9/28/16.
//  Copyright © 2016 ctang. All rights reserved.
//
//#include "opencv2/viz.hpp"
#include "GlobalReconstruction.h"
#include "KeyFrame.h"
#include "Drawer.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ceres/ceres.h"

#include "theia/sfm/global_pose_estimation/robust_rotation_estimator.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/types.h"
#include <vector>

const bool disable_loop=true;

namespace GSLAM {
    

    void GlobalReconstruction::getScaleConstraint(KeyFrame* keyFrame1,
                                                  vector<ScaleConstraint>& scaleConstraints){
        
        for(map<KeyFrame*,vector<int> >::iterator mit=keyFrame1->mConnectedKeyFrameMatches.begin(),
            mend=keyFrame1->mConnectedKeyFrameMatches.end();
            mit!=mend;
            mit++){
            
            KeyFrame* keyFrame2=mit->first;
            
            if (keyFrame2->mvLocalFrames.empty()) {
                continue;
            }
            
            Transform pose=keyFrame1->mConnectedKeyFramePoses[keyFrame2];
            ScaleConstraint constraint;
            constraint.keyFrameIndex1=keyFrame1->mnId;
            constraint.keyFrameIndex2=keyFrame2->mnId;
            vector<double> scales;
            vector<int>& matches=mit->second;
            
            for (int i=0;i<matches.size();i++) {
                if (matches[i]>=0) {
                    
                    assert(keyFrame1->mvLocalMapPoints[i].isEstimated);
                    if(!keyFrame2->mvLocalMapPoints[matches[i]].isEstimated){
                        continue;
                    }
                    
                    Eigen::Vector3d distance1=keyFrame1->mvLocalMapPoints[i].getPosition()-pose.translation;
                    Eigen::Vector3d distance2=keyFrame2->mvLocalMapPoints[matches[i]].getPosition();
                    
                    /*printf("%d %d %x %x\n",keyFrame1->outId,keyFrame2->outId,
                           keyFrame1->mvLocalMapPoints[i].gMP,keyFrame2->mvLocalMapPoints[matches[i]].gMP);
                    
                    for(map<KeyFrame*,int>::iterator mit=keyFrame1->mvLocalMapPoints[i].gMP->measurements.begin(),
                        mend=keyFrame1->mvLocalMapPoints[i].gMP->measurements.end();
                        mit!=mend; mit++){
                        printf("%d %d\n",mit->first->outId,mit->second);
                    }
                    
                    for(map<KeyFrame*,int>::iterator mit=keyFrame2->mvLocalMapPoints[matches[i]].gMP->measurements.begin(),
                        mend=keyFrame2->mvLocalMapPoints[matches[i]].gMP->measurements.end();
                        mit!=mend; mit++){
                        printf("%d %d\n",mit->first->outId,mit->second);
                    }*/
                    
                    //assert(keyFrame1->mvLocalMapPoints[i].gMP==keyFrame2->mvLocalMapPoints[matches[i]].gMP);
                    double relativeScale=distance2.norm()/distance1.norm();
                    scales.push_back(relativeScale);
                }
            }
            
            if (scales.size()>=scaleThreshold) {
                sort(scales.begin(),scales.end());
                constraint.value12=log(scales[scales.size()/2]);
            }else{
                continue;
            }
            
            constraint.weight=1.0;
            scaleConstraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::getRotationConstraint(KeyFrame* keyFrame1,
                                                     vector<RotationConstraint>& rotationConstraints){
        
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
            mend=keyFrame1->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            
            KeyFrame* keyFrame2=mit->first;
            Transform pose=keyFrame1->mConnectedKeyFramePoses[keyFrame2];
            RotationConstraint constraint;
            constraint.keyFrameIndex1=keyFrame1->mnId;
            constraint.keyFrameIndex2=keyFrame2->mnId;
            constraint.rotation12=pose.rotation;
            if(pose.scale==-1&&(keyFrame1->frameId<70||keyFrame2->frameId<70)){
                continue;
            }
            
            //printf("add %d %d\n",constraint.keyFrameIndex1,constraint.keyFrameIndex2);
            rotationConstraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::getTranslationConstraint(KeyFrame* keyFrame1,
                                                        vector<TranslationConstraint>& translationConstraints){
        
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
            mend=keyFrame1->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            KeyFrame* keyFrame2=mit->first;
            Transform pose=keyFrame1->mConnectedKeyFramePoses[keyFrame2];
            TranslationConstraint constraint;
            constraint.keyFrameIndex1=keyFrame1->mnId;
            constraint.keyFrameIndex2=keyFrame2->mnId;
            constraint.rotation1=keyFrame1->pose.rotation;
            constraint.translation12=pose.translation/keyFrame1->scale;
            if(pose.scale==-1&&(keyFrame1->frameId<70||keyFrame2->frameId<70)){
                continue;
            }
            translationConstraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::getSIM3Constraint(KeyFrame* keyFrame1,vector<SIM3Constraint>& sim3Constraints){
        
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
            mend=keyFrame1->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            KeyFrame* keyFrame2=mit->first;
            Transform pose=keyFrame1->mConnectedKeyFramePoses[keyFrame2];
            SIM3Constraint constraint;
            constraint.keyFrameIndex1=keyFrame1->mnId;
            constraint.keyFrameIndex2=keyFrame2->mnId;
            constraint.rotation12=pose.rotation;
            constraint.translation12=pose.translation;
            constraint.weight=1.0;
            
            vector<int>& matches=keyFrame1->mConnectedKeyFrameMatches[keyFrame2];
            vector<double> scales;
            for (int i=0;i<matches.size();i++) {
                if (matches[i]>=0) {
                    assert(keyFrame1->mvLocalMapPoints[i].isEstimated);
                    if(!keyFrame2->mvLocalMapPoints[matches[i]].isEstimated){
                        continue;
                    }
                    Eigen::Vector3d distance1=keyFrame1->mvLocalMapPoints[i].getPosition()-pose.translation;
                    Eigen::Vector3d distance2=keyFrame2->mvLocalMapPoints[matches[i]].getPosition();
                    double relativeScale=distance2.norm()/distance1.norm();
                    scales.push_back(relativeScale);
                }
            }
            
            if(scales.empty()){
                continue;
            }
            
            sort(scales.begin(),scales.end());
            constraint.scale12=scales[scales.size()/2];
            
            sim3Constraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::estimateScale(){
        
        vector<int> scaleIndex(keyFrames.size(),-1);
        vector<ScaleConstraint> scaleConstraints;
        
        for (int i=0;i<keyFrames.size();i++) {
            assert(keyFrames[i]->mnId==i);
            getScaleConstraint(keyFrames[i],scaleConstraints);
        }
        //printf("%d keyframe size\n",keyFrames.size());
        int nScales=0;
        for (int i=0;i<scaleConstraints.size();i++) {
            if (scaleIndex[scaleConstraints[i].keyFrameIndex1]==-1) {
                scaleIndex[scaleConstraints[i].keyFrameIndex1]=nScales;
                nScales++;
            }
            
            if (scaleIndex[scaleConstraints[i].keyFrameIndex2]==-1) {
                scaleIndex[scaleConstraints[i].keyFrameIndex2]=nScales;
                nScales++;
            }
            //printf("%d %d\n",scaleIndex[scaleConstraints[i].keyFrameIndex1],scaleIndex[scaleConstraints[i].keyFrameIndex2]);
            scaleConstraints[i].variableIndex1=scaleIndex[scaleConstraints[i].keyFrameIndex1];
            scaleConstraints[i].variableIndex2=scaleIndex[scaleConstraints[i].keyFrameIndex2];
        }
        
        //extrac constraint for first keyframe
        ScaleConstraint constraint;
        constraint.keyFrameIndex1=-1;
        constraint.keyFrameIndex2=0;
        constraint.variableIndex1=-1;
        constraint.variableIndex2=0;
        constraint.value1=0.0;
        constraint.value12=0.0;
        constraint.weight=1.0;
        scaleConstraints.push_back(constraint);
        
        vector<double> newScales;
        newScales.resize(nScales);
        for(int i=0;i<keyFrames.size();i++){
            if (scaleIndex[i]!=-1) {
                newScales[scaleIndex[i]]=keyFrames[i]->logScale;
            }
        }
        /*for (int i=0;i<scaleConstraints.size();i++) {
            
            cout<<scaleConstraints[i].keyFrameIndex1<<' '
                     <<scaleConstraints[i].keyFrameIndex2<<' '
                     <<scaleConstraints[i].variableIndex1<<' '
                     <<scaleConstraints[i].variableIndex2<<' '
                     <<scaleConstraints[i].value12<<endl;
        }
        for (int i=0;i<newScales.size();i++) {
            cout<<i<<' '<<newScales[i]<<endl;
        }*/
        globalScaleEstimation.maxIterations=10000;
        globalScaleEstimation.solve(scaleConstraints,newScales);
        
        for (int i=1;i<newScales.size();i++) {
            newScales[i]-=newScales[0];
        }
        newScales[0]=0.0;
        
        for (int i=0;i<keyFrames.size();i++) {
            keyFrames[i]->logScale=newScales[scaleIndex[i]];
            keyFrames[i]->scale=exp(keyFrames[i]->logScale);
            printf("%f\n",keyFrames[i]->scale);
        }
        
        static ofstream record("/Users/chaos/Desktop/debug/scales_error.txt");
        for (int i=0;i<scaleConstraints.size();i++) {
            if (scaleConstraints[i].keyFrameIndex1==-1) {
                continue;
            }
            double diff=keyFrames[scaleConstraints[i].keyFrameIndex2]->logScale-keyFrames[scaleConstraints[i].keyFrameIndex1]->logScale;
            diff=abs(diff-scaleConstraints[i].value12);
            int id1=keyFrames[scaleConstraints[i].keyFrameIndex1]->outId;
            int id2=keyFrames[scaleConstraints[i].keyFrameIndex2]->outId;
            //printf("%f %d %d\n",diff,id1,id2);
            record<<id1<<' '
                  <<id2<<' '
                  <<diff<<endl;
        }
        //getchar();
    }
    
    void GlobalReconstruction::estimateRotation(vector<int> &rotationIndex){
        
        std::vector<RotationConstraint> rotationConstraints(0);
        std::vector<Eigen::Matrix3d> newRotations;
        
        int nRotations=0;
        rotationIndex.resize(keyFrames.size(),-1);
        for (int k=0;k<keyFrames.size();k++) {
            
            getRotationConstraint(keyFrames[k],rotationConstraints);
            for (int i=0;i<rotationConstraints.size();i++) {
                
                if (rotationIndex[rotationConstraints[i].keyFrameIndex1]==-1) {
                    rotationIndex[rotationConstraints[i].keyFrameIndex1]=nRotations;
                    nRotations++;
                    newRotations.push_back(keyFrames[rotationConstraints[i].keyFrameIndex1]->pose.rotation);
                }
                
                if (rotationIndex[rotationConstraints[i].keyFrameIndex2]==-1) {
                    rotationIndex[rotationConstraints[i].keyFrameIndex2]=nRotations;
                    nRotations++;
                    newRotations.push_back(keyFrames[rotationConstraints[i].keyFrameIndex2]->pose.rotation);
                }
                
                
                rotationConstraints[i].variableIndex1=rotationIndex[rotationConstraints[i].keyFrameIndex1];
                rotationConstraints[i].variableIndex2=rotationIndex[rotationConstraints[i].keyFrameIndex2];
            }
        }
        
        std::unordered_map<theia::ViewIdPair,theia::TwoViewInfo> view_pairs;
        std::unordered_map<theia::ViewId,Eigen::Vector3d> orientations;
        
        theia::RobustRotationEstimator::Options options;
        options.max_num_irls_iterations=100;
        options.max_num_l1_iterations=10;
        theia::RobustRotationEstimator rotation_estimator(options);
        
        for (int i=0;i<rotationConstraints.size();i++) {
            theia::ViewIdPair viewPair;
            viewPair.first=rotationConstraints[i].variableIndex1;
            viewPair.second=rotationConstraints[i].variableIndex2;
            Eigen::Vector3d angle;
            ceres::RotationMatrixToAngleAxis(rotationConstraints[i].rotation12.data(),angle.data());
            view_pairs[viewPair].rotation_2=angle;
        }
        newRotations.resize(keyFrames.size());
        newRotations[0]=Eigen::Matrix3d::Identity();
        for (int i=1;i<keyFrames.size();i++) {
            newRotations[i]=keyFrames[i-1]->mvLocalFrames.back().pose.rotation*newRotations[i-1];
        }
        
        
        for (int i=0;i<keyFrames.size();i++) {
            Eigen::Vector3d angle;
            ceres::RotationMatrixToAngleAxis(newRotations[i].data(),angle.data());
            orientations[i]=angle;
        }
        std::cout<<"theia rotation start"<<std::endl;
        rotation_estimator.EstimateRotations(view_pairs,&orientations);
        std::cout<<"theia rotation finished"<<std::endl;
        
        for (int i=0;i<newRotations.size();i++) {
            ceres::AngleAxisToRotationMatrix(orientations[i].data(),newRotations[i].data());
            std::cout<<orientations[i].transpose()<<std::endl;
        }
        
        for (int i=1;i<newRotations.size();i++) {
            newRotations[i]=newRotations[i]*newRotations[0].transpose();
        }
        newRotations[0]=Eigen::Matrix3d::Identity();
        
        for (int k=0;k<keyFrames.size();k++) {
            keyFrames[k]->pose.rotation=newRotations[rotationIndex[k]];
        }
        std::cout<<"rotation end"<<std::endl;
    }
    
    void GlobalReconstruction::estimateRotationRobust(const vector<int> &rotationIndex){
        
        vector<RotationConstraint> rotationConstraints(0);
        vector<Eigen::Matrix3d> newRotations(keyFrames.size());

        for (int k=0;k<keyFrames.size();k++) {
            getRotationConstraint(keyFrames[k],rotationConstraints);
            for (int i=0;i<rotationConstraints.size();i++) {
                rotationConstraints[i].variableIndex1=rotationIndex[rotationConstraints[i].keyFrameIndex1];
                rotationConstraints[i].variableIndex2=rotationIndex[rotationConstraints[i].keyFrameIndex2];
            }
        }
        
        for (int i=0;i<newRotations.size();i++) {
            newRotations[i]=keyFrames[i]->pose.rotation;
        }
        
        RotationConstraint constraint;
        constraint.keyFrameIndex1=-1;
        constraint.keyFrameIndex2=0;
        constraint.variableIndex1=-1;
        constraint.variableIndex2=0;
        constraint.rotation1=Eigen::Matrix3d::Identity();
        constraint.rotation12=Eigen::Matrix3d::Identity();
        constraint.weight=1.0;
        rotationConstraints.push_back(constraint);
        
        
        globalRotationEstimation.maxOuterIterations=1000;
        globalRotationEstimation.maxInnerIterations=20;
        globalRotationEstimation.solve(rotationConstraints,newRotations);
        
        for (int i=1;i<newRotations.size();i++) {
            newRotations[i]=newRotations[i]*newRotations[0].transpose();
        }
        newRotations[0]=Eigen::Matrix3d::Identity();
        
        for (int k=0;k<keyFrames.size();k++) {
            keyFrames[k]->pose.rotation=newRotations[rotationIndex[k]];
        }
    }
    
    void GlobalReconstruction::estimateTranslation(const vector<int> &translationIndex){
        
        vector<TranslationConstraint> translationConstraints;
        for (int k=0;k<keyFrames.size();k++) {
            getTranslationConstraint(keyFrames[k],translationConstraints);
        }
        
        for (int i=0;i<translationConstraints.size();i++) {
            translationConstraints[i].variableIndex1=translationIndex[translationConstraints[i].keyFrameIndex1];
            translationConstraints[i].variableIndex2=translationIndex[translationConstraints[i].keyFrameIndex2];
        }
        
        TranslationConstraint constraint;
        constraint.keyFrameIndex1=-1;
        constraint.keyFrameIndex2=0;
        constraint.variableIndex1=-1;
        constraint.variableIndex2=0;
        constraint.rotation1=Eigen::Matrix3d::Identity();
        constraint.translation1=Eigen::Vector3d::Zero();
        constraint.translation12=Eigen::Vector3d::Zero();
        constraint.weight=1.0;
        translationConstraints.push_back(constraint);
        
        
        for (int i=0;i<translationConstraints.size();i++) {
            translationConstraints[i].translation12=translationConstraints[i].rotation1.transpose()
                                                   *translationConstraints[i].translation12;
            
            /*cout<<translationConstraints[i].variableIndex1<<' '
            <<translationConstraints[i].variableIndex2<<endl
            <<translationConstraints[i].translation12<<endl<<translationConstraints[i].rotation1<<endl;*/
        }
        
        vector<Eigen::Vector3d> newTranslations(keyFrames.size(),Eigen::Vector3d::Zero());
        globalTranslationEstimation.maxIterations=10000;
        globalTranslationEstimation.solve(translationConstraints,newTranslations);
        
        for (int i=1;i<newTranslations.size();i++) {
            newTranslations[i]-=newTranslations[0];
        }
        newTranslations[0]=Eigen::Vector3d::Zero();
        
        for (int k=0;k<keyFrames.size();k++) {
            keyFrames[k]->pose.translation=newTranslations[translationIndex[k]];
            cout<<keyFrames[k]->pose.translation.transpose()<<endl;
        }
        
        static ofstream record("/Users/chaos/Desktop/debug/trans_error.txt");
        
        
        for (int i=0;i<translationConstraints.size();i++) {
            if (translationConstraints[i].keyFrameIndex1==-1) {
                continue;
            }
            
            Eigen::Vector3d differror=translationConstraints[i].translation12-(keyFrames[translationConstraints[i].keyFrameIndex2]->pose.translation-keyFrames[translationConstraints[i].keyFrameIndex1]->pose.translation);
            
            double diff=differror.norm();
            
            int id1=keyFrames[translationConstraints[i].keyFrameIndex1]->outId;
            int id2=keyFrames[translationConstraints[i].keyFrameIndex2]->outId;
            
            record<<id1<<' '
                  <<id2<<' '
                  <<diff<<endl;

        }
    }
    struct GlobalError2{
        
        GlobalError2(const double _tracked_x,const double _tracked_y):
        tracked_x(_tracked_x), tracked_y(_tracked_y){
        }
        
        template <typename T>
        bool operator()(const T* const camera,
                        const T* const point,
                        T* residuals) const {
            
            
            
            T transformed[3];
            ceres::AngleAxisRotatePoint(camera,point,transformed);
            
            transformed[0]+=camera[3];
            transformed[1]+=camera[4];
            transformed[2]+=camera[5];
            
            
            T predicted_x=transformed[0]/transformed[2];
            T predicted_y=transformed[1]/transformed[2];
            
            residuals[0]=predicted_x-(T)tracked_x;
            residuals[1]=predicted_y-(T)tracked_y;
            
            residuals[0]=residuals[0];
            residuals[1]=residuals[1];
            
            return true;
        }
        
        static ceres::CostFunction* Create(const double tracked_x,
                                           const double tracked_y) {
            return (new ceres::AutoDiffCostFunction<GlobalError2,2,6,3>(new GlobalError2(tracked_x,tracked_y)));
        };
        
        double tracked_x;
        double tracked_y;
    };
    void GlobalReconstruction::globalRefine(){
        
        std::vector<double>             points;
        std::vector<double>             cameras;
        std::vector<std::pair<int,int>> pairCameraPoint;
        
        int nPoints=0;
        cameras.resize(6*keyFrames.size());
        for (int i=0;i<keyFrames.size();i++) {
            KeyFrame* keyFrame=keyFrames[i];
            
            Eigen::Vector3d translation=-keyFrame->pose.rotation*keyFrame->pose.translation;
            ceres::RotationMatrixToAngleAxis(keyFrame->pose.rotation.data(),
                                             &cameras[6*i]);
            cameras[6*i+3]=translation(0);
            cameras[6*i+4]=translation(1);
            cameras[6*i+5]=translation(2);
            
            nPoints+=keyFrame->mvLocalMapPoints.size();
        }
        points.resize(3*nPoints);
        nPoints=0;
        
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.003);
        
        
        for (int k=0;k<keyFrames.size();k++) {
            
            KeyFrame* keyFrame1=keyFrames[k];
            assert(keyFrame1->mnId==k);
            for (int i=0;i<keyFrame1->mvLocalMapPoints.size();i++) {
                if (keyFrame1->mvLocalMapPoints[i].isEstimated) {
                    
                    Eigen::Vector3d point3D=keyFrame1->mvLocalMapPoints[i].getPosition();
                    point3D/=keyFrame1->scale;
                    point3D=keyFrame1->pose.rotation.transpose()*point3D+keyFrame1->pose.translation;
                    
                    points[3*(nPoints+i)]=point3D(0);
                    points[3*(nPoints+i)+1]=point3D(1);
                    points[3*(nPoints+i)+2]=point3D(2);
                }
            }
            std::vector<bool> status(keyFrame1->mvLocalMapPoints.size(),false);
            for(std::map<KeyFrame*,std::vector<int> >::iterator mit=keyFrame1->mConnectedKeyFrameMatches.begin(),
                mend=keyFrame1->mConnectedKeyFrameMatches.end();
                mit!=mend;
                mit++){
                
                KeyFrame* keyFrame2=mit->first;
                if (keyFrame2->mvLocalFrames.empty()) {
                    continue;
                }
                Transform transform=keyFrame1->mConnectedKeyFramePoses[keyFrame2];
                
                if (transform.scale<-10.0&&disable_loop) {
                    continue;
                }
                
                std::vector<int>& matches=mit->second;
                assert(matches.size()==keyFrame1->mvLocalMapPoints.size());
                
                for (int i=0;i<matches.size();i++) {
                    if (matches[i]>=0) {
                        
                        assert(keyFrame1->mvLocalMapPoints[i].isEstimated);
                        
                        Eigen::Vector3d projection(&points[3*(nPoints+i)]);
                        projection=keyFrame2->pose.rotation*(projection-keyFrame2->pose.translation);
                        projection/=projection(2);
                        projection-=keyFrame2->mvLocalMapPoints[matches[i]].vec;
                        
                        
                        if(!keyFrame2->mvLocalMapPoints[matches[i]].isEstimated&&projection.norm()>0.008){
                            continue;
                        }
                        ceres::CostFunction* cost_function = GlobalError2::Create(keyFrame2->mvLocalMapPoints[matches[i]].vec(0),
                                                                                  keyFrame2->mvLocalMapPoints[matches[i]].vec(1));
                        problem.AddResidualBlock(cost_function,loss_function,
                                                 &cameras[6*keyFrame2->mnId],
                                                 &points[3*(nPoints+i)]);
                        status[i]=true;
                    }
                }
            }
            
            if(keyFrame1->nextKeyFramePtr!=NULL){
                for(int i=0;i<keyFrame1->mvLocalMapPoints.size();i++){
                    
                    Eigen::Vector3d projection(&points[3*(nPoints+i)]);
                    projection=keyFrame1->nextKeyFramePtr->pose.rotation*(projection-keyFrame1->nextKeyFramePtr->pose.translation);
                    projection/=projection(2);
                    projection-=(*keyFrame1->mvLocalMapPoints[i].vecs.back());
                    
                    if (keyFrame1->mvLocalMapPoints[i].isEstimated
                        &&keyFrame1->mvLocalMapPoints[i].vecs.back()!=NULL&&projection.norm()<0.008) {
                        assert(keyFrame1->mvLocalMapPoints[i].vecs.size()==keyFrame1->mvLocalFrames.size());
                        Eigen::Vector3d projection=(*keyFrame1->mvLocalMapPoints[i].vecs.back());
                        projection/=projection(2);
                        
                        ceres::CostFunction* cost_function = GlobalError2::Create(projection(0),projection(1));
                        problem.AddResidualBlock(cost_function,loss_function,
                                                 &cameras[6*keyFrame1->nextKeyFramePtr->mnId],
                                                 &points[3*(nPoints+i)]);
                        status[i]=true;
                    }
                }
            }
            
            for (int i=0;i<keyFrame1->mvLocalMapPoints.size();i++) {
                if (status[i]==true) {
                    
                    ceres::CostFunction* cost_function = GlobalError2::Create(keyFrame1->mvLocalMapPoints[i].vec(0),
                                                                              keyFrame1->mvLocalMapPoints[i].vec(1));
                    problem.AddResidualBlock(cost_function,loss_function,
                                             &cameras[6*keyFrame1->mnId],
                                             &points[3*(nPoints+i)]);
                    keyFrame1->mvLocalMapPoints[i].isEstimated=true;
                }else{
                    keyFrame1->mvLocalMapPoints[i].isEstimated=false;
                }
            }
            nPoints+=keyFrame1->mvLocalMapPoints.size();
        }
        
        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations=100;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        for(int i=0;i<keyFrames.size();i++){
            
            Eigen::Matrix3d rotation;
            ceres::AngleAxisToRotationMatrix(&cameras[6*i],
                                             rotation.data());
            
            Eigen::Vector3d translation;
            translation(0)=cameras[6*i+3];
            translation(1)=cameras[6*i+4];
            translation(2)=cameras[6*i+5];
            translation=-rotation.transpose()*translation;
            
            keyFrames[i]->pose.rotation=rotation;
            keyFrames[i]->pose.translation=translation;
        }
        
        
        
        nPoints=0;
        for (int k=0;k<keyFrames.size();k++) {
            std::vector<double> scales;
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if (keyFrames[k]->mvLocalMapPoints[i].isEstimated) {
                    keyFrames[k]->mvLocalMapPoints[i].globalPosition(0)=points[3*(nPoints+i)];
                    keyFrames[k]->mvLocalMapPoints[i].globalPosition(1)=points[3*(nPoints+i)+1];
                    keyFrames[k]->mvLocalMapPoints[i].globalPosition(2)=points[3*(nPoints+i)+2];
                    
                    Eigen::Vector3d globalDistance=keyFrames[k]->mvLocalMapPoints[i].globalPosition
                    -keyFrames[k]->pose.translation;
                    Eigen::Vector3d localDistance=keyFrames[k]->mvLocalMapPoints[i].getPosition();
                    scales.push_back(localDistance.norm()/globalDistance.norm());
                }
            }
            std::sort(scales.begin(),scales.end());
            keyFrames[k]->scale=scales[scales.size()/2];
            nPoints+=keyFrames[k]->mvLocalMapPoints.size();
        }
        
        for(int i=1;i<keyFrames.size();i++){
            keyFrames[i]->pose.rotation=keyFrames[i]->pose.rotation*keyFrames[0]->pose.rotation.transpose();
            keyFrames[i]->pose.translation=keyFrames[0]->pose.rotation*(keyFrames[i]->pose.translation
                                                                        -keyFrames[0]->pose.translation);
        }
        
        for (int k=0;k<keyFrames.size();k++) {
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if (keyFrames[k]->mvLocalMapPoints[i].isEstimated) {
                    keyFrames[k]->mvLocalMapPoints[i].globalPosition=keyFrames[0]->pose.rotation*
                    (keyFrames[k]->mvLocalMapPoints[i].globalPosition-keyFrames[0]->pose.translation);
                }
            }
        }
        
        keyFrames[0]->pose.rotation=Eigen::Matrix3d::Identity();
        keyFrames[0]->pose.translation=Eigen::Vector3d::Zero();
    }
    
    
    
    void GlobalReconstruction::addNewKeyFrame(KeyFrame* keyFrame){
        keyFrames.push_back(keyFrame);
    }
    
    void GlobalReconstruction::savePly(){
        
        ofstream results(path+std::string("/keyframes.txt"));
        for (int k=0;k<keyFrames.size();k++) {
            results<<keyFrames[k]->frameId<<' '<<keyFrames[k]->pose.translation.transpose();
            for (int i1=0;i1<3;i1++) {
                for (int i2=0;i2<3;i2++) {
                    results<<' '<<keyFrames[k]->pose.rotation(i1,i2);
                }
            }
            results<<endl;
        }
        results.close();
        
        
        ofstream trajectories(path+std::string("/frames.txt"));
        for (int k=0;k<keyFrames.size();k++) {
            for (int i=0;i<keyFrames[k]->mvLocalFrames.size();i++) {
                
                Eigen::Matrix3d rotation;
                rotation=keyFrames[k]->mvLocalFrames[i].pose.rotation*keyFrames[k]->pose.rotation;

                
                Eigen::Vector3d translation=keyFrames[k]->mvLocalFrames[i].pose.translation;
                translation=keyFrames[k]->pose.rotation.transpose()*translation;
                translation/=keyFrames[k]->scale;
                translation+=keyFrames[k]->pose.translation;
                
                trajectories<<keyFrames[k]->mvLocalFrames[i].frameId<<' '<<translation.transpose();
                for (int i1=0;i1<3;i1++) {
                    for (int i2=0;i2<3;i2++) {
                        trajectories<<' '<<rotation(i1,i2);
                    }
                }
                trajectories<<endl;
            }
        }
        trajectories.close();
        
        int npoints=0;
        for (int k=0;k<keyFrames.size();k++) {
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                keyFrames[k]->mvLocalMapPoints[i].isEstimated=false;
            }
            
            for(map<KeyFrame*,vector<int> >::iterator mit=keyFrames[k]->mConnectedKeyFrameMatches.begin(),
                mend=keyFrames[k]->mConnectedKeyFrameMatches.end();
                mit!=mend;
                mit++){
                
                for (int i=0;i<mit->second.size();i++) {
                    if (mit->second[i]>=0) {
                        keyFrames[k]->mvLocalMapPoints[i].isEstimated=true;
                        npoints++;
                    }
                }
            }
        }
        
        ofstream pointcloud(path+std::string("/pointcloud.ply"));
        pointcloud << "ply"
        << '\n' << "format ascii 1.0"
        << '\n' << "element vertex " <<npoints
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        << '\n' << "end_header" << std::endl;
        
        for (int nFrame=0;nFrame<keyFrames.size();nFrame++) {
            char name[200];
            for(int k=0;k<=nFrame;k++){
                for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                    if(keyFrames[k]->mvLocalMapPoints[i].isEstimated){
                        
                        Eigen::Vector3d point3D=keyFrames[k]->mvLocalMapPoints[i].getPosition();
                        point3D/=keyFrames[k]->scale;
                        point3D=keyFrames[k]->pose.rotation.transpose()*point3D+keyFrames[k]->pose.translation;
                        
                        uchar* color=keyFrames[k]->mvLocalMapPoints[i].color;
                        pointcloud <<point3D.transpose()<<' '<<(int)color[2]<<' '<<(int)color[1]<<' '<<(int)color[0]<<endl;
                    }
                }
            }
        }
        pointcloud.close();
    }
    

    
    void GlobalReconstruction::visualize(){
        
        pangolin::WindowInterface& window=pangolin::CreateWindowAndBind("GSLAM: Map Viewer",720,720);
        //window.Move(100,480);
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
                                          pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ,0,0,0,0.0,-1.0, 0.0));
        
        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
        
        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();
        
        bool bFollow = true;
        bool bLocalizationMode = false;
        float mT = 1e3/30.0;
        Drawer drawer;
        drawer.mCameraSize=0.08;
        drawer.mCameraLineWidth=3;
        drawer.mFrameIndex=0;
        drawer.mKeyFrameIndex=0;
        drawer.keyFrames=keyFrames;
        drawer.mGraphLineWidth=3.0;
        
//        cv::namedWindow("track");
//        cv::moveWindow("track",720,0);
        //getchar();
        
        drawer.preTwc.resize(1);
        drawer.preTwc[0];
        drawer.preTwc[0].m[12]=0.0;
        drawer.preTwc[0].m[13]=0.0;
        drawer.preTwc[0].m[14]=0.0;
        cv::Mat preImage;
        while(1){
            
            if(drawer.mFrameIndex>=keyFrames.back()->mvLocalFrames.back().frameId){
                break;
            }
            
            cv::Mat image;
            if(drawer.mFrameIndex<keyFrames.back()->mvLocalFrames.back().frameId){
                drawer.getCurrentOpenGLCameraMatrix(Twc);
                
//                char name[200];
//                sprintf(name,"%s/1080/frame%05d.pgm",path,drawer.mFrameIndex+frameStart+1);
//                printf(name);
//                image=cv::imread(name);
//                printf("image %d %d\n",image.cols,image.rows);
//                if ((drawer.mFrameIndex)==keyFrames[drawer.mKeyFrameIndex]->frameId) {
//                    cv::RNG rng(-1);
//                    for(int i=0;i<keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints.size();i++){
//                        if(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount<=0){
//                            continue;
//                        }
//                        //printf("%f %f %f\n",color.val[0],color.val[1],color[2]);
//                        uchar *color=keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].color;
//
//                        int Size=min(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount,20);
//                        Size=max(Size,3);
//                        cv::circle(image,
//                                   keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[0],Size,
//                                   CV_RGB(245,211,40),2,CV_AA);
//
//                    }
//                }else{
//                    for(int i=0;i<keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints.size();i++){
//
//                        if(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount<=0){
//                            continue;
//                        }
//
//
//                        if((drawer.mFrameIndex-keyFrames[drawer.mKeyFrameIndex]->frameId)
//                           >=keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked.size()){
//                            continue;
//                        }
//                        int localIndex=drawer.mFrameIndex-keyFrames[drawer.mKeyFrameIndex]->frameId;
//
//                        int Size=min(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount,20);
//                        Size=max(Size,3);
//                        uchar *color=keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].color;
//                        cv::circle(image,keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[localIndex],Size,
//                                   CV_RGB(245,211,40),2,CV_AA);
//
//                        for (int j=localIndex;j>max(localIndex-5,0);j--) {
//                            cv::line(image,
//                                     keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[j],
//                                     keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[j-1],
//                                     CV_RGB(245,211,40),2,CV_AA);
//                        }
//                    }
//                }
//                cv::resize(image,image,cv::Size(720,405));
//                image.copyTo(preImage);
                s_cam.Follow(Twc);
            }else{
                preImage.copyTo(image);
                s_cam.Follow(Twc);
            }
            
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);
            
            drawer.drawCurrentCamera(Twc);
            drawer.drawKeyFrames();
            drawer.drawPoints();
        
//            char name[200];
//            sprintf(name,"%s/viewer/viewr%05d",path,drawer.mFrameIndex);
//            pangolin::SaveWindowOnRender(name);
//            sprintf(name,"%s/viewer/track%05d.png",path,drawer.mFrameIndex);
//            cv::imwrite(name,image);
//            image.release();
            
            pangolin::FinishFrame();
        }
    }
    bool myfunction (const LocalMapPoint* p1,const LocalMapPoint* p2) { return (p1->invdepth<p2->invdepth); }
    
    void GlobalReconstruction::topview(){
        
        /*for (int k=0;k<keyFrames.size();k++) {
            vector<LocalMapPoint*> localMapPoints;
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if (keyFrames[k]->mvLocalMapPoints[i].isEstimated) {
                    localMapPoints.push_back(&keyFrames[k]->mvLocalMapPoints[i]);
                }
            }
            sort(localMapPoints.begin(),localMapPoints.end(),myfunction);
            
            for (int i=0;i<(0.8*localMapPoints.size());i++) {
                localMapPoints[i]->isEstimated=false;
            }
        }*/
        //getchar();
        //pangolin::WindowInterface& window=pangolin::CreateWindowAndBind("GSLAM: Map Viewer",720,720);
        //window.Move(100,480);
        // 3D Mouse handler requires depth testing to be enabled
        //glEnable(GL_DEPTH_TEST);
        
        // Issue specific OpenGl we might need
        //glEnable (GL_BLEND);
        //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        //pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
        
        float mViewpointX=0;
        float mViewpointY=0;
        float mViewpointZ=-1.8;
        float mViewpointF=500;
        
        for(int i=0;i<keyFrames.size();i++){
            mViewpointX+=keyFrames[i]->pose.translation(0);
            mViewpointY+=keyFrames[i]->pose.translation(1);
        }
        mViewpointX/=keyFrames.size();
        mViewpointY/=keyFrames.size();
        
        
        float maxdistance=0.0;
        for (int i=1;i<keyFrames.size();i++) {
            float distance=(float)keyFrames[i]->pose.translation.norm();
            if (distance>maxdistance) {
                maxdistance=distance;
            }
        }
        
        printf("%f %f\n",mViewpointX,mViewpointY);
        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                                          pangolin::ModelViewLookAt(0,
                                                                    -maxdistance,
                                                                    0,
                                                                    0,0.0,0,
                                                                    pangolin::AxisDirection::AxisZ));
        
        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
        
        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();
        
        bool bFollow = true;
        bool bLocalizationMode = false;
        float mT = 1e3/30.0;
        Drawer drawer;
        drawer.mCameraSize=0.5;
        drawer.mCameraLineWidth=3;
        drawer.mFrameIndex=0;
        drawer.mKeyFrameIndex=0;
        drawer.keyFrames=keyFrames;
        drawer.mGraphLineWidth=5.0;
        
        cv::namedWindow("track");
        cv::moveWindow("track",720,0);
        
        
        drawer.preTwc.resize(1);
        drawer.preTwc[0];
        drawer.preTwc[0].m[12]=0.0;
        drawer.preTwc[0].m[13]=0.0;
        drawer.preTwc[0].m[14]=0.0;
        
        pangolin::OpenGlMatrix curretMatrix;
        while(1){
            
            if(drawer.mFrameIndex<keyFrames.back()->mvLocalFrames.back().frameId){
                drawer.getCurrentOpenGLCameraMatrix(Twc);
            }else{
                break;
            }
            
            curretMatrix=s_cam.GetModelViewMatrix();

            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);
            
            drawer.drawCurrentCamera(Twc);
            drawer.drawKeyFrames();
            drawer.drawPoints();
            
            char name[200];
            //sprintf(name,"%s/viewer/top%05d",path,drawer.mFrameIndex);
            //pangolin::SaveWindowOnRender(name);
            pangolin::FinishFrame();
            
        }
        
        drawer.mFrameIndex=0;
        drawer.mKeyFrameIndex=0;
        drawer.preTwc.resize(1);
        drawer.preTwc[0];
        drawer.preTwc[0].m[12]=0.0;
        drawer.preTwc[0].m[13]=0.0;
        drawer.preTwc[0].m[14]=0.0;
        while(1){
            
            if(drawer.mFrameIndex<keyFrames.back()->mvLocalFrames.back().frameId){
                drawer.getCurrentOpenGLCameraMatrix(Twc);
            }
            
            s_cam.SetModelViewMatrix(curretMatrix);
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);
            
            drawer.drawCurrentCamera(Twc);
            drawer.drawKeyFrames();
            drawer.drawPoints();
            
//            char name[200];
//            sprintf(name,"%s/viewer/top%05d",path,drawer.mFrameIndex);
//            pangolin::SaveWindowOnRender(name);
            pangolin::FinishFrame();
            
        }
    }
}
