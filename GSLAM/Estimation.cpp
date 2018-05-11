//
//  Estimation.cpp
//  GSLAM
//
//  Created by ctang on 9/10/16.
//  Copyright © 2016 ctang. All rights reserved.
//
#include "svdlib.hpp"
#include "svdutil.hpp"
#include "Estimation.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <iostream>
#include "Geometry.h"
#include "RelativeMotion.hpp"
#include "opencv2/calib3d/calib3d.hpp"


bool getPVector(const Eigen::Vector3d &T12,
                const Eigen::Vector3d &p1,
                const Eigen::Vector3d &p2,
                Eigen::Vector3d       &pVector){
    
    Eigen::Vector3d T=p1;
    Eigen::Vector3d r1=T12;
    Eigen::Vector3d r2=-p2;
    Eigen::Vector3d xij=r1.cross(r2);
    
    //analytic solution
    Eigen::Matrix3d A;
    for(int i=0;i<3;i++){
        A(i,0)=r1(i);
        A(i,1)=-r2(i);
        A(i,2)=-xij(i);
    }
    double determinantA=A.determinant();
    
    Eigen::Matrix3d D=A;
    D.col(0)=T;
    double determinant1=D.determinant();
    
    A.col(1)=T;
    double determinant2=A.determinant();
    
    double s_ik=determinant1/determinantA;
    double s_jk=determinant2/determinantA;
    
    if(s_ik<0||s_jk<0){
        return false;
    }
    
    Eigen::Matrix3d R_jik=ComputeAxisRotation(T,r1);
    Eigen::Matrix3d R_ijk=ComputeAxisRotation(-T,r2);
    
    Eigen::Matrix3d PMatrix=(Eigen::Matrix3d::Identity()+s_ik*R_jik-s_jk*R_ijk);
    pVector=PMatrix*p1;
    pVector=0.5*pVector;
    return true;
}


void getScaleRatio(const Eigen::Vector3d &T12,
                   const Eigen::Vector3d &p1,
                   const Eigen::Vector3d &p2,
                   double &s1,double &s2){
    
    Eigen::Vector3d T=p1;
    Eigen::Vector3d r1=T12;
    Eigen::Vector3d r2=-p2;
    Eigen::Vector3d xij=r1.cross(r2);
    
    //analytic solution
    Eigen::Matrix3d A;
    for(int i=0;i<3;i++){
        A(i,0)=r1(i);
        A(i,1)=-r2(i);
        A(i,2)=-xij(i);
    }
    double determinantA=A.determinant();
    
    Eigen::Matrix3d D=A;
    D.col(0)=T;
    double determinant1=D.determinant();
    
    A.col(1)=T;
    double determinant2=A.determinant();
    
    s1=determinant1/determinantA;
    s2=determinant2/determinantA;
}



Eigen::Vector3d estimateRelativeTranslationIRLS(const std::vector<Eigen::Vector3d> &pts1,
                                                const std::vector<Eigen::Vector3d> &pts2,
                                                std::vector<Eigen::Vector3d> &pVectors,
                                                std::vector<bool>            &status){
    int k = 0;
    int kmax = 100;
    int max_in_iter = 10;
    int in_iter = 0;
    double epslon = 1e-5;
    
    
    int num_point=pts1.size();
    Eigen::Matrix3Xd allNorms=Eigen::Matrix3Xd(3,num_point);
    for(int i=0;i<num_point;i++){
        allNorms.col(i)=pts1[i].cross(pts2[i]);
        allNorms.col(i).normalize();
    }
    
    Eigen::VectorXd weights=Eigen::VectorXd::Ones(num_point);
    double costVal = 0.0;
    Eigen::Vector3d translation=Eigen::Vector3d::Random();
    translation.normalize();
    
    while((in_iter < max_in_iter)&&(k < kmax)){
        Eigen::Matrix3Xd allNormsWeighted=Eigen::Matrix3Xd(3,num_point);
        for(int i=0;i<num_point;i++){
            double weight=1.0/std::max(weights(i),1e-5);
            allNormsWeighted.col(i)=weight*allNorms.col(i);
        }
        
        Eigen::Matrix3d NtN=allNormsWeighted*allNormsWeighted.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(NtN,Eigen::ComputeFullV);
        Eigen::Matrix3d V=svd.matrixV();
        Eigen::Vector3d translationNew=V.col(2);
        
        Eigen::VectorXd weightsNew=(translationNew.transpose()*allNorms);
        weights=weightsNew.cwiseAbs();
        
        double costNew=weights.sum();
        double delt = std::max(std::abs(costVal-costNew),1-translationNew.dot(translation));
        
        if (delt <= epslon){
            in_iter++;
        }
        else{
            in_iter=0;
        }
        
        costVal=costNew;
        translation=translationNew;
        k++;
        break;
    }
    
    pVectors.resize(num_point);
    status.resize(num_point);
    std::fill(status.begin(),status.end(),true);
    
    std::vector<double> s1(num_point),s2(num_point);
    for (int i=0;i<num_point;i++) {
        getScaleRatio(translation,pts1[i],pts2[i],s1[i],s2[i]);
    }
    
    int overallCount=0;
    int positiveCount=0;
    
    for (int i=0;i<num_point;i++) {
        if(s1[i]*s2[i]<0){
            status[i]=false;
            continue;
        }
        overallCount++;
        positiveCount+=s1[i]>0;
    }
    
    bool    isPositive=positiveCount>overallCount/2;
    double  factor=isPositive?1.0:-1.0;
    translation*=factor;
    
    for (int i=0;i<num_point;i++) {
        if (status[i]==true) {
            if ((s1[i]>0)==isPositive) {
                
                double s_1=factor*s1[i];
                double s_2=factor*s2[i];
                
                Eigen::Matrix3d R_jik=ComputeAxisRotation(pts1[i],translation);
                Eigen::Matrix3d R_ijk=ComputeAxisRotation(-pts1[i],-pts2[i]);
                
                Eigen::Matrix3d PMatrix=(Eigen::Matrix3d::Identity()+s_1*R_jik-s_2*R_ijk);
                pVectors[i]=0.5*PMatrix*pts1[i];
            }else{
                status[i]=false;
            }
        }
    }
    return translation;
}


Eigen::Vector3d estimateRelativeTranslation(const std::vector<Eigen::Vector3d> &pts1,
                                            const std::vector<Eigen::Vector3d> &pts2,
                                            std::vector<Eigen::Vector3d> &pVectors,
                                            std::vector<bool>            &status){
    
    int num_point=pts1.size();
    std::vector<Eigen::Vector3d> norms(num_point);
    pVectors.resize(num_point);
    status.resize(num_point);
    
    Eigen::Matrix3Xd allNorms=Eigen::Matrix3Xd(3,num_point);
    Eigen::Vector3d preResult,curResult;
    for(int i=0;i<num_point;i++){
        norms[i]=pts1[i].cross(pts2[i]);
        norms[i].normalize();
        allNorms.col(i)=norms[i];
    }

    Eigen::Matrix3d NtN=allNorms*allNorms.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(NtN,Eigen::ComputeFullV);
    Eigen::Matrix3d V=svd.matrixV();
    curResult=V.col(2);
    
    int positiveCount=0;
    std::vector<int> flags(num_point);
    for(int i=0;i<num_point;i++){
        Eigen::Vector3d flowVector=pts1[i]-pts2[i];
        flags[i]=(flowVector.dot(curResult)>0);
        positiveCount+=flags[i];
    }
    int flag=1;
    if((double)positiveCount<0.5*(double)num_point){
        curResult*=-1;
        flag=0;
    }
    
    for(int i=0;i<num_point;i++){
        if(flags[i]!=flag){
            continue;
        }
        status[i]=getPVector(curResult,pts1[i],pts2[i],pVectors[i]);
    }
    return curResult;
}




typedef struct {
    bool flag;
    double value;
    int    index;
}Residual;

typedef struct{
    bool operator() (const Residual &r1,const Residual &r2) {
        return r1.value<r2.value;
    }
}compareResidual;

Eigen::Vector3d estimateRelativeTranslationRobust(const std::vector<Eigen::Vector3d> &pts1,
                                                  const std::vector<Eigen::Vector3d> &pts2,
                                                  std::vector<bool>            &status,
                                                  const double inlierPercentage=0.8,
                                                  const double inlierThreshold=std::sin((3.0/180.0)*CV_PI)){
    
    int num_point=pts1.size();
    std::vector<Residual> residuals(num_point);
    std::vector<Eigen::Vector3d> norms(num_point);
    status.resize(num_point);
    std::vector<Eigen::Vector3d> pVectors(num_point);
    
    Eigen::Matrix3Xd allNorms=Eigen::Matrix3Xd(3,num_point);
    
    
    
    int numInliers=0;
    double preError=1.0,curError=1.0;
    Eigen::Vector3d preResult,curResult;
    
    for(int iter=0;iter<10;iter++){
        if(iter==0){
            for(int i=0;i<num_point;i++){
                norms[i]=pts1[i].cross(pts2[i]);
                norms[i].normalize();
                allNorms.col(i)=norms[i];
            }
            Eigen::Matrix3d NtN=allNorms*allNorms.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(NtN,Eigen::ComputeFullV);
            Eigen::Matrix3d V=svd.matrixV();
            curResult=V.col(2);
            Eigen::VectorXd errors=curResult.transpose()*allNorms;
            
            for(int i=0;i<num_point;i++){
                residuals[i].index=i;
                residuals[i].value=std::abs(errors(i));
            }
        }else{
            
            Eigen::Matrix3Xd subNorms=Eigen::Matrix3Xd(3,numInliers);
            for(int i=0;i<numInliers;i++){
                int index=residuals[i].index;
                subNorms.col(i)=norms[index];
            }
            Eigen::Matrix3d NtN=subNorms*subNorms.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(NtN,Eigen::ComputeFullV);
            Eigen::Matrix3d V=svd.matrixV();
            curResult=V.col(2);
            Eigen::VectorXd errors=curResult.transpose()*allNorms;
            for(int i=0;i<num_point;i++){
                int index=residuals[i].index;
                residuals[i].value=std::abs(errors(index));
            }
        }
        std::sort(residuals.begin(),residuals.end(),compareResidual());
        
        int medInliers=residuals.size()*inlierPercentage;
        curError=residuals[medInliers].value;
        
        if(curError<preError){
            preResult=curResult;
            preError=curError;
            numInliers=medInliers;
        }else{
            break;
        }
        
        while(numInliers<residuals.size()&&residuals[numInliers].value<inlierThreshold){
            numInliers++;
        }
    }
    
    std::fill(status.begin(),status.end(),false);
    
    int positiveCount=0;
    for(int i=0;i<numInliers;i++){
        int index=residuals[i].index;
        Eigen::Vector3d flowVector=pts1[index]-pts2[index];
        residuals[i].flag=(flowVector.dot(preResult)>0);
        positiveCount+=residuals[i].flag;
    }
    
    int flag=1;
    if((double)positiveCount<0.5*(double)numInliers){
        preResult*=-1;
        flag=0;
    }
    
    for(int i=0;i<numInliers;i++){
        if(residuals[i].flag!=flag){
            continue;
        }
        int index=residuals[i].index;
        status[index]=getPVector(preResult,pts1[index],pts2[index],pVectors[index]);
    }
    return preResult;
}

Eigen::Vector3d estimateRelativeTranslationRobust(const std::vector<Eigen::Vector3d> &pts1,
                                                  const std::vector<Eigen::Vector3d> &pts2,
                                                  std::vector<Eigen::Vector3d> &pVectors,
                                                  std::vector<bool>            &status,
                                                  std::vector<double>          &errors,
                                                  double                       &medError,
                                                  const double inlierPercentage=0.8,
                                                  const double inlierThreshold=std::sin((3.0/180.0)*CV_PI)){
    
    int num_point=pts1.size();
    std::vector<Residual> residuals(num_point);
    std::vector<Eigen::Vector3d> norms(num_point);
    pVectors.resize(num_point);
    status.resize(num_point);
    errors.resize(num_point);
    
    Eigen::Matrix3Xd allNorms=Eigen::Matrix3Xd(3,num_point);
    
    
    
    int numInliers=0;
    double preError=1.0,curError=1.0;
    Eigen::Vector3d preResult,curResult;
    
    for(int iter=0;iter<10;iter++){
        if(iter==0){
            for(int i=0;i<num_point;i++){
                norms[i]=pts1[i].cross(pts2[i]);
                norms[i].normalize();
                allNorms.col(i)=norms[i];
            }
            Eigen::Matrix3d NtN=allNorms*allNorms.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(NtN,Eigen::ComputeFullV);
            Eigen::Matrix3d V=svd.matrixV();
            curResult=V.col(2);
            Eigen::VectorXd errors=curResult.transpose()*allNorms;
            
            for(int i=0;i<num_point;i++){
                residuals[i].index=i;
                residuals[i].value=std::abs(errors(i));
            }
        }else{
            
            Eigen::Matrix3Xd subNorms=Eigen::Matrix3Xd(3,numInliers);
            for(int i=0;i<numInliers;i++){
                int index=residuals[i].index;
                subNorms.col(i)=norms[index];
            }
            Eigen::Matrix3d NtN=subNorms*subNorms.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(NtN,Eigen::ComputeFullV);
            Eigen::Matrix3d V=svd.matrixV();
            curResult=V.col(2);
            Eigen::VectorXd errors=curResult.transpose()*allNorms;
            for(int i=0;i<num_point;i++){
                int index=residuals[i].index;
                residuals[i].value=std::abs(errors(index));
            }
        }
        std::sort(residuals.begin(),residuals.end(),compareResidual());
        
        int medInliers=residuals.size()*inlierPercentage;
        curError=residuals[medInliers].value;
        
        if(curError<preError){
            preResult=curResult;
            preError=curError;
            numInliers=medInliers;
        }else{
            break;
        }
        
        while(numInliers<residuals.size()&&residuals[numInliers].value<inlierThreshold){
            numInliers++;
        }
    }
    
    for(int i=0;i<num_point;i++){
        errors[residuals[i].index]=residuals[i].value;
    }
    std::fill(status.begin(),status.end(),false);
    
    int positiveCount=0;
    for(int i=0;i<numInliers;i++){
        int index=residuals[i].index;
        Eigen::Vector3d flowVector=pts1[index]-pts2[index];
        residuals[i].flag=(flowVector.dot(preResult)>0);
        positiveCount+=residuals[i].flag;
    }
    
    int flag=1;
    if((double)positiveCount<0.5*(double)numInliers){
        preResult*=-1;
        flag=0;
    }
    
    for(int i=0;i<numInliers;i++){
        if(residuals[i].flag!=flag){
            continue;
        }
        int index=residuals[i].index;
        status[index]=getPVector(preResult,pts1[index],pts2[index],pVectors[index]);
    }
    
    medError=curError;
    return preResult;
}


void estimateRelativeMotion(const std::vector<Eigen::Vector3d> &pts1,
                            std::vector<Eigen::Vector3d> &pts2,
                            Eigen::Matrix3d &rotation,
                            Eigen::Vector3d &translation,
                            std::vector<Eigen::Vector3d> &pVectors,
                            std::vector<bool>            &status){
    
    
    estimateRotationTranslation(0.005,
                                pts1,
                                pts2,
                                rotation,
                                translation);
    
    //estimateRotationTranslation2(0.005,pts1,pts2,rotation,translation);
    translation=rotation.transpose() * translation;
    translation.normalize();
    
    
    for (int i=0;i<pts2.size();i++) {
        pts2[i]=rotation.transpose()*pts2[i];
        pts2[i].normalize();
    }
    int num_point=pts1.size();
    int positiveCount=0;
    std::vector<int> flags(num_point);
    for(int i=0;i<num_point;i++){
        Eigen::Vector3d flowVector=pts1[i]-pts2[i];
        flags[i]=(flowVector.dot(translation)>0);
        positiveCount+=flags[i];
    }
    int flag=1;
    if((double)positiveCount<0.5*(double)num_point){
        translation*=-1;
        flag=0;
    }
    
    status.resize(num_point);
    std::fill(status.begin(),status.end(),false);
    pVectors.resize(num_point);
    
    for(int i=0;i<num_point;i++){
        if(flags[i]!=flag){
            continue;
        }
        status[i]=getPVector(translation,pts1[i],pts2[i],pVectors[i]);
    }
}



struct SnavelyReprojectionError {
    
    SnavelyReprojectionError(const double _key_x,const double _key_y,
                             const double _tracked_x,const double _tracked_y,const double _weight):
    key_x(_key_x),key_y(_key_y),
    tracked_x(_tracked_x), tracked_y(_tracked_y),
    weight(_weight){
    }
    
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const depth,
                    T* residuals) const {
        
        T p[3];
        T point[3]={((T)key_x)/depth[0],((T)key_y)/depth[0],1.0/depth[0]};
        ceres::AngleAxisRotatePoint(camera,point,p);
        
        p[0]+=camera[3];
        p[1]+=camera[4];
        p[2]+=camera[5];
        
        T predicted_x=p[0]/p[2];
        T predicted_y=p[1]/p[2];
        
        residuals[0]=predicted_x-(T)tracked_x;
        residuals[1]=predicted_y-(T)tracked_y;
        
        residuals[0]=residuals[0]*weight;
        residuals[1]=residuals[1]*weight;
        
        return true;
    }
    
    static ceres::CostFunction* Create(const double key_x,
                                       const double key_y,
                                       const double tracked_x,
                                       const double tracked_y,
                                       const double weight=1.0) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,6,1>(new SnavelyReprojectionError(key_x,key_y,tracked_x,tracked_y,weight)));
    };
    
    double tracked_x;
    double tracked_y;
    double key_x;
    double key_y;
    double weight;
};


struct SnavelyReprojectionErrorSimple {
    
    SnavelyReprojectionErrorSimple(
                                   const double _key_x,const double _key_y,
                                   const double _tracked_x,const double _tracked_y):
    key_x(_key_x),key_y(_key_y),
    tracked_x(_tracked_x), tracked_y(_tracked_y){
    }
    
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const depth,
                    T* residuals) const {
        
        T ax=key_x-camera[2]*key_y+camera[1];
        T bx=camera[3];
        T ay=key_y-camera[0]+camera[2]*key_x;
        T by=camera[4];
        T c =-camera[1]*key_x+camera[0]*key_y+1.0;
        T d =camera[5];
        T ex=tracked_x*c-ax;
        T ey=tracked_y*c-ay;
        
        if(depth[0]<0.01){
            residuals[0]=ex;
            residuals[1]=ey;
            return true;
        }
        
        T fx=tracked_x*d-bx;
        T fy=tracked_y*d-by;
        
        residuals[0]=(ex+fx*depth[0])/(c+d*depth[0]);
        residuals[1]=(ey+fy*depth[0])/(c+d*depth[0]);
        return true;
    }
    
    static ceres::CostFunction* Create(
                                       const double key_x,
                                       const double key_y,
                                       const double tracked_x,
                                       const double tracked_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorSimple,2,6,1>(new SnavelyReprojectionErrorSimple(key_x,key_y,tracked_x,tracked_y)));
    }
    
    double tracked_x;
    double tracked_y;
    double key_x;
    double key_y;
};

void setSmallRotationMatrix(const double *w,Eigen::Matrix3d &rotation){
    rotation(0,0)=rotation(1,1)=rotation(2,2)=1.0;
    rotation(0,1)=-w[2];rotation(0,2)=w[1];
    rotation(1,0)=w[2];rotation(1,2)=-w[0];
    rotation(2,0)=-w[1];rotation(2,1)=w[0];
}

namespace GSLAM{
    
    int RelativeOutlierRejection::relativeRansac(const KeyFrame *keyFrame,const KLT_FeatureList featureList){
        
        assert(keyFrame->mvLocalMapPoints.size()==featureList->nFeatures);
        
        vector<cv::Point2d> points1,points2;
        vector<int> indices;
        points1.reserve(featureList->nFeatures);
        points2.reserve(featureList->nFeatures);
        indices.reserve(featureList->nFeatures);
        
        int medViewAngleCount=0,overallCount=0;
        Eigen::Matrix3d rotationTranspoed=rotation.transpose();
        
        
        for (int i=0;i<featureList->nFeatures;i++) {
            if(featureList->feature[i]->val==KLT_TRACKED&&(*isValid)[i]){
                
                overallCount++;
                
                Eigen::Vector3d p1=keyFrame->mvLocalMapPoints[i].norm;
                Eigen::Vector3d p2=rotationTranspoed*featureList->feature[i]->norm;
                p2.normalize();
                double viewAngle=p1.dot(p2);
                
                if (viewAngle>=minViewAngle) {
                    (*isValid)[i]=false;
                    continue;
                }
                
                if (viewAngle<medViewAngle) {
                    medViewAngleCount++;
                }
                
                p1=K*p1;
                p2=K*p2;
                
                points1.push_back(cv::Point2d(p1(0)/p1(2),p1(1)/p1(2)));
                points2.push_back(cv::Point2d(p2(0)/p2(2),p2(1)/p2(2)));
                indices.push_back(i);
            }
        }
        
        if (medViewAngleCount<0.5*overallCount) {
            return -1;
        }
        
        points1.shrink_to_fit();
        points2.shrink_to_fit();
        indices.shrink_to_fit();
        
        vector<uchar> status;
        cv::findFundamentalMat(points1,points2,cv::FM_RANSAC,theshold,prob,status);
        
        int inlierCount=status.size();
        for (int i=0;i<status.size();i++) {
            if (status[i]==0) {
                int index=indices[i];
                (*isValid)[index]=false;
                inlierCount--;
            }
        }
        return inlierCount;
    }
    

    
    int RelativePoseEstimation::estimateRelativePose(const KeyFrame *keyFrame,
                                                     const KLT_FeatureList featureList,
                                                     std::vector<Eigen::Vector3d*> &pVectors){
        
        
        std::vector<double> viewAngles(keyFrame->mvLocalMapPoints.size());
        std::fill(viewAngles.begin(),viewAngles.end(),-1.0);
        
        pVectors.resize(keyFrame->mvLocalMapPoints.size());
        std::fill(pVectors.begin(),pVectors.end(),static_cast<Eigen::Vector3d*>(NULL));
    
        std::vector<Eigen::Vector3d> norms1,norms2;
        std::vector<int> indices;
        
        indices.reserve(keyFrame->mvLocalMapPoints.size());
        norms1.reserve(keyFrame->mvLocalMapPoints.size());
        norms2.reserve(keyFrame->mvLocalMapPoints.size());
        
        
        
        for (int i=0;i<keyFrame->mvLocalMapPoints.size();i++) {
            if (featureList->feature[i]->val==KLT_TRACKED&&(*isValid)[i]) {
                Eigen::Vector3d norm2=rotation.transpose()*featureList->feature[i]->norm;
                norm2.normalize();
                norms1.push_back(keyFrame->mvLocalMapPoints[i].norm);
                norms2.push_back(norm2);
                indices.push_back(i);
            }
        }
        norms1.shrink_to_fit();
        norms2.shrink_to_fit();
        indices.shrink_to_fit();
        
        std::vector<bool> status;
        std::vector<Eigen::Vector3d> tmpPVectors;
        translation=estimateRelativeTranslation(norms1,norms2,tmpPVectors,status);
        
        //translation=estimateRelativeTranslation(norms1,norms2,tmpPVectors,status);
        //Eigen::Matrix3d inputRotation;
        //Eigen::Vector3d inputTranslation;
        
        //this->rotation=Eigen::Matrix3d::Identity();
        //this->translation=Eigen::Vector3d::Zero();
        
        //estimateRelativeMotion(norms1,norms2,this->rotation,this->translation,tmpPVectors,status);
        
        /*Eigen::Matrix3d transposedRotation=rotation.transpose();
        for (int i=0;i<norms2.size();i++) {
            norms2[i]=transposedRotation*norms2[i];
        }
        translation=estimateRelativeTranslation(norms1,norms2,tmpPVectors,status);*/
        
        
        
        int successCount=0;
        for (int i=0;i<status.size();i++) {
            int ind=indices[i];
            if (status[i]==true) {
                pVectors[ind]=new Eigen::Vector3d();
                *pVectors[ind]= tmpPVectors[i];
                successCount++;
            }else{
                //printf("%d %d\n",ind,i);
            }
        }
        return successCount;
    }
    
    
    void LocalFactorization::process(KeyFrame* keyFrame){
        
        std::vector<int> indices(0);
        for(int i=0;i<keyFrame->mvLocalMapPoints.size();i++){
            keyFrame->mvLocalMapPoints[i].isFullTrack&=(!keyFrame->mvLocalMapPoints[i].vecs.empty());
            if(keyFrame->mvLocalMapPoints[i].isFullTrack){
                indices.push_back(i);
            }
        }
        
        int num_frame=keyFrame->mvLocalFrames.size();
        int num_point=indices.size();
        
        //printf("%d %d\n",num_frame,num_point);getchar();
        
        DMat denseMatrix=svdNewDMat(num_point,3*num_frame);
        for(int p=0;p<num_point;p++){
            int ind=indices[p];
            const std::vector<Eigen::Vector3d*>& pVectors=keyFrame->mvLocalMapPoints[ind].pVectors;
            for(int f=0;f<num_frame;f++){
                memcpy(&denseMatrix->value[p][3*f],&(*pVectors[f])(0),3*sizeof(double));
            }
        }
        SMat sparseMatrix=svdConvertDtoS(denseMatrix);
        SVDRec svdResult=svdLAS2A(sparseMatrix,12);
        
#ifdef DEBUG
        printf("singular %f %f\n",svdResult->S[0],svdResult->S[1]);
#endif
        
        
        svdFreeDMat(denseMatrix);
        svdFreeSMat(sparseMatrix);
        
        
        int positiveCount=0;
        for(int p=0;p<num_point;p++){
            positiveCount+=(svdResult->Ut->value[0][p]>0);
        }
        
        if(positiveCount<num_point/2){
            positiveCount=-1;
        }else{
            positiveCount=1;
        }
        
        //convert results to 3d coordinates
        for(int p=0;p<num_point;p++){
            svdResult->Ut->value[0][p]*=positiveCount;
            int ind=indices[p];
            if (svdResult->Ut->value[0][p]>0) {
                keyFrame->mvLocalMapPoints[ind].invdepth=svdResult->Ut->value[0][p]/keyFrame->mvLocalMapPoints[ind].norm(2);
                keyFrame->mvLocalMapPoints[ind].isEstimated=true;
            }
        }
        
        for(int f=0;f<num_frame;f++){
            keyFrame->mvLocalFrames[f].pose.translation=Eigen::Vector3d(&svdResult->Vt->value[0][3*f]);
            keyFrame->mvLocalFrames[f].pose.translation*=positiveCount*svdResult->S[0];
        }
    }
    
    void LocalFactorization::iterativeRefine(KeyFrame *keyFrame){
        
        std::vector<int> indices(0);
        for(int i=0;i<keyFrame->mvLocalMapPoints.size();i++){
            keyFrame->mvLocalMapPoints[i].isFullTrack&=(!keyFrame->mvLocalMapPoints[i].vecs.empty());
            if(keyFrame->mvLocalMapPoints[i].isFullTrack){
                indices.push_back(i);
            }
        }
        std::vector<cv::Point2d> vecs1(indices.size());
        std::vector<cv::Point2d> vecs2(indices.size());
        
        /*for (int i=0;i<vecs0.size();i++) {
            Eigen::Vector3d vec=keyFrame->mvLocalMapPoints[indices[i]].vec;
            vecs0[i].x=vec(0);
            vecs0[i].y=vec(1);
        }*/
        
        
        std::ofstream of("/Users/ctang/Desktop/debug/res.txt");
        //double converge=0.002;
        for (int f=keyFrame->mvLocalFrames.size()-5;f<keyFrame->mvLocalFrames.size()-4;f++) {
            for (int i=0;i<indices.size();i++) {
                Eigen::Vector3d point3d=keyFrame->mvLocalMapPoints[indices[i]].getPosition();
                Eigen::Vector3d vec1=keyFrame->mvLocalFrames[f].pose.rotation*(point3d-keyFrame->mvLocalFrames[f].pose.translation);
                vec1/=vec1(2);
                Eigen::Vector3d vec2=*keyFrame->mvLocalMapPoints[indices[i]].vecs[f];
                vecs1[i].x=vec1(0);
                vecs1[i].y=vec1(1);
                
                vecs2[i].x=vec2(0);
                vecs2[i].y=vec2(1);
            }
            std::vector<uchar> statuts;
            cv::Mat homography=cv::findHomography(vecs1,vecs2,CV_RANSAC,0.005,statuts);
            std::vector<cv::Mat> rotation(1),translation(1),normal(1);
            cv::Mat calib=cv::Mat::eye(3,3,CV_64FC1);
            cv::decomposeHomographyMat(homography,calib,
                                       rotation,translation,normal);
            /*printf("%d\n",f);
            for (int i1=0;i1<3;i1++) {
                for (int i2=0;i2<3;i2++) {
                    printf("%f ",rotation[0].at<double>(i1,i2));
                }
                printf("\n");
            }*/
            //printf("%d\n",rotation.size());
            //getchar();
            for (int i=0;i<indices.size();i++) {
                if (statuts[i]==0) {
                    continue;
                }
                cv::Mat vec1(3,1,CV_64FC1);
                vec1.at<double>(0)=vecs1[i].x;
                vec1.at<double>(1)=vecs1[i].y;
                vec1.at<double>(2)=1.0;
                
                double diffx=vec1.at<double>(0)-vecs2[i].x;
                double diffy=vec1.at<double>(1)-vecs2[i].y;
                
                of<<sqrt(diffx*diffx+diffy*diffy)<<' ';
                for (int j=0;j<4;j++) {
                    cv::Mat _vec1=rotation[j]*vec1;
                    _vec1/=_vec1.at<double>(2);
                    diffx=_vec1.at<double>(0)-vecs2[i].x;
                    diffy=_vec1.at<double>(1)-vecs2[i].y;
                    of<<sqrt(diffx*diffx+diffy*diffy)<<' ';
                }
                of<<endl;
            }
            getchar();
        }
        of.close();
    }
    
    void LocalBundleAdjustment::bundleAdjust(KeyFrame *keyFrame,bool cameraFixed){
        
        int num_point=0;
        std::vector<int> indices;
        
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            if (keyFrame->mvLocalMapPoints[p].isEstimated) {
                indices.push_back(p);
                num_point++;
            }
        }
        
        int num_frame=keyFrame->mvLocalFrames.size();
        std::vector<int> pointNumberInFrames(num_frame,num_point);
        
        std::vector<double> param_cam(6*num_frame),param_invDepth(num_point);
        ceres::Problem problem;
        printf("%d %d\n",num_point,num_frame);
        
        for(int p=0;p<num_point;p++){
            int index=indices[p];
            param_invDepth[p]=keyFrame->mvLocalMapPoints[index].invdepth;
        }
        
        for(int f=0;f<num_frame;f++){
            ceres::RotationMatrixToAngleAxis(&keyFrame->mvLocalFrames[f].pose.rotation(0),&param_cam[6*f]);
            Eigen::Vector3d translation=keyFrame->mvLocalFrames[f].pose.rotation*keyFrame->mvLocalFrames[f].pose.translation;
            param_cam[6*f+3]=-translation(0);
            param_cam[6*f+4]=-translation(1);
            param_cam[6*f+5]=-translation(2);
        }
        
        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.005);
        
        for(int p=0;p<num_point;p++){
            int index=indices[p];
            Eigen::Vector3d vec1=keyFrame->mvLocalMapPoints[index].vec;
            for(int f=0;f<num_frame;f++){
                
                if(keyFrame->mvLocalMapPoints[index].vecs[f]==NULL){
                    continue;
                }
                
                Eigen::Vector3d vec2=*keyFrame->mvLocalMapPoints[index].vecs[f];
                ceres::CostFunction* cost_function =SnavelyReprojectionError::Create(vec1(0),vec1(1),vec2(0),vec2(1));
                problem.AddResidualBlock(cost_function,loss_function,&param_cam[6*f],&param_invDepth[p]);
            }
        }
        if (cameraFixed) {
            for(int f=0;f<num_frame;f++){
                problem.SetParameterBlockConstant(&param_cam[6*f]);
            }
        }
        
        
        ceres::Solver::Options options;
        options.num_threads=1;
        options.max_num_iterations=5;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.logging_type=ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        for(int f=0;f<num_frame;f++){
            //printf("%d\n",f);
            //Eigen::Matrix3d rotation=keyFrame->mvLocalFrames[f].pose.rotation;
            ceres::AngleAxisToRotationMatrix(&param_cam[6*f],&keyFrame->mvLocalFrames[f].pose.rotation(0));
            //rotation=rotation.transpose()*keyFrame->mvLocalFrames[f].pose.rotation;
            //std::cout<<rotation<<endl;
            
            keyFrame->mvLocalFrames[f].pose.translation=-Eigen::Vector3d(&param_cam[6*f+3]);
            keyFrame->mvLocalFrames[f].pose.translation=keyFrame->mvLocalFrames[f].pose.rotation.transpose()
                                                       *keyFrame->mvLocalFrames[f].pose.translation;
            keyFrame->mvLocalFrames[f].pose.setEssentialMatrix();
        }
        
        for(int p=0;p<num_point;p++){
            int index=indices[p];
            keyFrame->mvLocalMapPoints[index].invdepth=param_invDepth[p];
            keyFrame->mvLocalMapPoints[index].updateNormalAndDepth();
        }
        
        for(int p=0;p<num_point;p++){
            
            int index=indices[p];
            Eigen::Vector3d point3D=keyFrame->mvLocalMapPoints[index].getPosition();
            
            if(keyFrame->mvLocalMapPoints[index].errors.empty()){
                keyFrame->mvLocalMapPoints[index].errors.resize(num_frame);
            }
            
            for (int f=0;f<num_frame;f++) {
                
                if(keyFrame->mvLocalMapPoints[index].vecs[f]==NULL){
                    continue;
                }
                
                Eigen::Vector3d projected=keyFrame->mvLocalFrames[f].pose.rotation
                *(point3D-keyFrame->mvLocalFrames[f].pose.translation);
                projected/=projected(2);
                projected-=(*keyFrame->mvLocalMapPoints[index].vecs[f]);
                double error=projected.norm();
                keyFrame->mvLocalMapPoints[index].errors[f]=error;
                
                if (error>projErrorThres) {
                    delete keyFrame->mvLocalMapPoints[index].vecs[f];
                    keyFrame->mvLocalMapPoints[index].vecs[f]=NULL;
                    keyFrame->mvLocalMapPoints[index].measurementCount--;
                    keyFrame->mvLocalFrames[f].measurementCount--;
                }
            }
            
            //assert(keyFrame->mvLocalMapPoints[index].measurementCount>0);
            if (keyFrame->mvLocalMapPoints[index].measurementCount==0||keyFrame->mvLocalMapPoints[index].invdepth<0) {
                keyFrame->mvLocalMapPoints[index].isEstimated=false;
            }
        }
        
        /*for (int i=0;i<keyFrame->mvLocalMapPoints.size();i++) {
            
        }*/
        
        int measurementCountByPoint=0,measurementCountByFrame=0;
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            measurementCountByPoint+=keyFrame->mvLocalMapPoints[p].measurementCount;
        }
        
        for (int f=0;f<keyFrame->mvLocalFrames.size();f++) {
            measurementCountByFrame+=keyFrame->mvLocalFrames[f].measurementCount;
        }
        printf("%d %d\n",measurementCountByPoint,measurementCountByFrame);
        assert(measurementCountByPoint==measurementCountByFrame);
    }
    
    
    
    void LocalBundleAdjustment::triangulate(KeyFrame* keyFrame){
        
        int num_frame=keyFrame->mvLocalFrames.size();
        
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            
            if(!keyFrame->mvLocalMapPoints[p].isEstimated&&keyFrame->mvLocalMapPoints[p].isValid()){
                
                std::vector<Eigen::Vector3d> normalized;
                std::vector<Eigen::Matrix3d> rotations;
                std::vector<Eigen::Vector3d> cameras;
                
                normalized.reserve(num_frame);
                rotations.reserve(num_frame);
                cameras.reserve(num_frame);
                
                normalized.push_back(keyFrame->mvLocalMapPoints[p].norm);
                rotations.push_back(Eigen::Matrix3d::Identity());
                cameras.push_back(Eigen::Vector3d::Zero());
                
                for (int f=0;f<num_frame;f++) {
                    
                    if (keyFrame->mvLocalMapPoints[p].vecs[f]==NULL) {
                        continue;
                    }
                    Eigen::Vector3d vec2=*keyFrame->mvLocalMapPoints[p].vecs[f];
                    Eigen::Vector3d norm2=keyFrame->mvLocalFrames[f].pose.rotation.transpose()*vec2;
                    norm2.normalize();
                    
                    double viewAngle=keyFrame->mvLocalMapPoints[p].norm.dot(norm2);
                    
                    if (viewAngle>viewAngleThres) {
                        
                        delete keyFrame->mvLocalMapPoints[p].vecs[f];
                        keyFrame->mvLocalMapPoints[p].vecs[f]=NULL;
                        
                        keyFrame->mvLocalMapPoints[p].measurementCount--;
                        keyFrame->mvLocalFrames[f].measurementCount--;
                        
                    }else{
                        
                        Eigen::Vector3d epiolarLine=keyFrame->mvLocalFrames[f].pose.E*keyFrame->mvLocalMapPoints[p].vec;
                        double squareError=epiolarLine.dot(vec2);
                        squareError*=squareError;
                        squareError/=(epiolarLine(0)*epiolarLine(0)+epiolarLine(1)*epiolarLine(1));
                        
                        if (squareError>4*projErrorThres*projErrorThres) {
                            
                            delete keyFrame->mvLocalMapPoints[p].vecs[f];
                            keyFrame->mvLocalMapPoints[p].vecs[f]=NULL;
                            
                            keyFrame->mvLocalMapPoints[p].measurementCount--;
                            keyFrame->mvLocalFrames[f].measurementCount--;
                            
                        }else{
                            normalized.push_back(norm2);
                            rotations.push_back(Eigen::Matrix3d::Identity());
                            cameras.push_back(keyFrame->mvLocalFrames[f].pose.translation);
                        }
                    }
                }
                
                normalized.shrink_to_fit();
                rotations.shrink_to_fit();
                cameras.shrink_to_fit();
                
                if (cameras.size()>1) {
                    
                    Eigen::Vector3d point3D=multiviewTriangulationLinear(normalized,rotations,cameras);
                    
                    bool isOutlier=false;
                    for(int i=1;i<cameras.size();i++){
                        
                        Eigen::Vector3d project=point3D-cameras[i];
                        project/=project(2);
                        
                        Eigen::Vector3d error=project-normalized[i]/normalized[i](2);
                        if (error.norm()>2*projErrorThres) {
                            isOutlier=true;
                            break;
                        }
                    }
                    
                    if (point3D(2)<0) {
                        isOutlier=true;
                    }
                    
                    if (!isOutlier) {
                        keyFrame->mvLocalMapPoints[p].isEstimated=true;
                        keyFrame->mvLocalMapPoints[p].invdepth=1.0/point3D(2);
                    }else{
                        for (int f=0;f<num_frame;f++) {
                            if (keyFrame->mvLocalMapPoints[p].vecs[f]!=NULL) {
                                delete keyFrame->mvLocalMapPoints[p].vecs[f];
                                keyFrame->mvLocalMapPoints[p].vecs[f]=NULL;
                                keyFrame->mvLocalFrames[f].measurementCount--;
                            }
                        }
                        keyFrame->mvLocalMapPoints[p].measurementCount=0;
                    }
                }
            }
        }
        //zheli
        
        int measurementCountByPoint=0,measurementCountByFrame=0;
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            measurementCountByPoint+=keyFrame->mvLocalMapPoints[p].measurementCount;
        }
        
        for (int f=0;f<keyFrame->mvLocalFrames.size();f++) {
            measurementCountByFrame+=keyFrame->mvLocalFrames[f].measurementCount;
        }
        assert(measurementCountByPoint==measurementCountByFrame);
        
        int effectiveCount=0;
        for(int i=0;i<keyFrame->mvLocalMapPoints.size();i++){
            effectiveCount+=keyFrame->mvLocalMapPoints[i].isValid()==true;
        }
        printf("%d\n",effectiveCount);
    }
    
    
    
    //refine points by projection error
    void LocalBundleAdjustment::refinePoints(KeyFrame *keyFrame){
        
        int num_point=0;
        std::vector<int> indices;
        
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            if (keyFrame->mvLocalMapPoints[p].isEstimated) {
                indices.push_back(p);
                num_point++;
            }
        }
        int num_frame=keyFrame->mvLocalFrames.size();
        
        
        for(int p=0;p<num_point;p++){
            
            int index=indices[p];
            Eigen::Vector3d point3D=keyFrame->mvLocalMapPoints[index].getPosition();
            
            if(keyFrame->mvLocalMapPoints[index].errors.empty()){
                keyFrame->mvLocalMapPoints[index].errors.resize(num_frame);
            }
            
            for (int f=0;f<num_frame;f++) {
                
                if(keyFrame->mvLocalMapPoints[index].vecs[f]==NULL){
                    continue;
                }
                
                Eigen::Vector3d projected=keyFrame->mvLocalFrames[f].pose.rotation
                *(point3D-keyFrame->mvLocalFrames[f].pose.translation);
                projected/=projected(2);
                projected-=(*keyFrame->mvLocalMapPoints[index].vecs[f]);
                double error=projected.norm();
                keyFrame->mvLocalMapPoints[index].errors[f]=error;
                
                if (error>projErrorThres) {
                    delete keyFrame->mvLocalMapPoints[index].vecs[f];
                    keyFrame->mvLocalMapPoints[index].vecs[f]=NULL;
                    keyFrame->mvLocalMapPoints[index].measurementCount--;
                    keyFrame->mvLocalFrames[f].measurementCount--;
                }
            }
            
            if (keyFrame->mvLocalMapPoints[index].measurementCount==0
              ||keyFrame->mvLocalMapPoints[index].invdepth<0) {
                keyFrame->mvLocalMapPoints[index].isEstimated=false;
            }
        }
        
        
        int measurementCountByPoint=0,measurementCountByFrame=0;
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            measurementCountByPoint+=keyFrame->mvLocalMapPoints[p].measurementCount;
        }
        
        for (int f=0;f<keyFrame->mvLocalFrames.size();f++) {
            measurementCountByFrame+=keyFrame->mvLocalFrames[f].measurementCount;
        }
        printf("%d %d\n",measurementCountByPoint,measurementCountByFrame);
        assert(measurementCountByPoint==measurementCountByFrame);
    }
    
    
    
    void LocalBundleAdjustment::triangulate2(KeyFrame* keyFrame){
        
        int num_frame=keyFrame->mvLocalFrames.size();
        
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            
            if(!keyFrame->mvLocalMapPoints[p].isEstimated&&keyFrame->mvLocalMapPoints[p].isValid()){
                
                std::vector<Eigen::Vector3d> normalized;
                std::vector<Eigen::Matrix3d> rotations;
                std::vector<Eigen::Vector3d> cameras;
                
                normalized.reserve(num_frame);
                rotations.reserve(num_frame);
                cameras.reserve(num_frame);
                
                normalized.push_back(keyFrame->mvLocalMapPoints[p].norm);
                rotations.push_back(Eigen::Matrix3d::Identity());
                cameras.push_back(Eigen::Vector3d::Zero());
                
                for (int f=0;f<num_frame;f++) {
                    
                    if (keyFrame->mvLocalMapPoints[p].vecs[f]==NULL) {
                        continue;
                    }
                    Eigen::Vector3d vec2=*keyFrame->mvLocalMapPoints[p].vecs[f];
                    Eigen::Vector3d norm2=keyFrame->mvLocalFrames[f].pose.rotation.transpose()*vec2;
                    norm2.normalize();
                    
                    double viewAngle=keyFrame->mvLocalMapPoints[p].norm.dot(norm2);
                    
                    if (viewAngle>viewAngleThres) {
                        
                        delete keyFrame->mvLocalMapPoints[p].vecs[f];
                        keyFrame->mvLocalMapPoints[p].vecs[f]=NULL;
                        
                        keyFrame->mvLocalMapPoints[p].measurementCount--;
                        keyFrame->mvLocalFrames[f].measurementCount--;
                        
                    }else{
                        
                        Eigen::Vector3d epiolarLine=keyFrame->mvLocalFrames[f].pose.E
                                                   *keyFrame->mvLocalMapPoints[p].vec;
                        
                        double squareError=epiolarLine.dot(vec2);
                        squareError*=squareError;
                        squareError/=(epiolarLine(0)*epiolarLine(0)+epiolarLine(1)*epiolarLine(1));
                        
                        if (squareError>4*projErrorThres*projErrorThres) {
                            
                            delete keyFrame->mvLocalMapPoints[p].vecs[f];
                            keyFrame->mvLocalMapPoints[p].vecs[f]=NULL;
                            
                            keyFrame->mvLocalMapPoints[p].measurementCount--;
                            keyFrame->mvLocalFrames[f].measurementCount--;
                            
                        }else{
                            normalized.push_back(norm2);
                            rotations.push_back(Eigen::Matrix3d::Identity());
                            cameras.push_back(keyFrame->mvLocalFrames[f].pose.translation);
                        }
                    }
                }
                
                normalized.shrink_to_fit();
                rotations.shrink_to_fit();
                cameras.shrink_to_fit();
                
                if (cameras.size()>1) {
                    
                    Eigen::Vector3d point3D=multiviewTriangulationLinear(normalized,rotations,cameras);
                    //double depth=multiviewTriangulationDepth(normalized,rotations,cameras);
                    //std::cout<<depth<<std::endl;
                    //getchar();
                    //Eigen::Vector3d point3D=depth*normalized[0];
                    
                    bool isOutlier=false;
                    for(int i=1;i<cameras.size();i++){
                        
                        Eigen::Vector3d project=point3D-cameras[i];
                        project/=project(2);
                        
                        Eigen::Vector3d error=project-normalized[i]/normalized[i](2);
                        if (error.norm()>2*projErrorThres) {
                            isOutlier=true;
                            break;
                        }
                    }
                    
                    if (point3D(2)<0) {
                        isOutlier=true;
                    }
                    
                    if (!isOutlier) {
                        keyFrame->mvLocalMapPoints[p].isEstimated=true;
                        keyFrame->mvLocalMapPoints[p].invdepth=1.0/point3D(2);
                    }else{
                        for (int f=0;f<num_frame;f++) {
                            if (keyFrame->mvLocalMapPoints[p].vecs[f]!=NULL) {
                                delete keyFrame->mvLocalMapPoints[p].vecs[f];
                                keyFrame->mvLocalMapPoints[p].vecs[f]=NULL;
                                keyFrame->mvLocalFrames[f].measurementCount--;
                            }
                        }
                        keyFrame->mvLocalMapPoints[p].measurementCount=0;
                    }
                }
            }
        }
        
        int measurementCountByPoint=0,measurementCountByFrame=0;
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            measurementCountByPoint+=keyFrame->mvLocalMapPoints[p].measurementCount;
        }
        
        for (int f=0;f<keyFrame->mvLocalFrames.size();f++) {
            measurementCountByFrame+=keyFrame->mvLocalFrames[f].measurementCount;
        }
        assert(measurementCountByPoint==measurementCountByFrame);
        
        int effectiveCount=0;
        for(int i=0;i<keyFrame->mvLocalMapPoints.size();i++){
            effectiveCount+=keyFrame->mvLocalMapPoints[i].isValid()==true;
        }
        printf("%d\n",effectiveCount);
    }

    
    /*void LocalBundleAdjustment::estimateFarFrames(KeyFrame* keyFrame){
        
        for (int i=0;i<keyFrame->mvFarFrames.size();i++) {
            
            std::vector<Eigen::Vector3d*>& closeMeasurements=keyFrame->mvFarMeasurements[i];
            
            std::vector<cv::Point3f> points3d;
            std::vector<cv::Point2f> points2d;
            
            for (int m=0;m<closeMeasurements.size();m++) {
                if (closeMeasurements[m]!=NULL&&keyFrame->mvLocalMapPoints[m].measurementCount>0) {
                    
                    cv::Point3f point3d;
                    cv::Point2f point2d;
                    Eigen::Vector3d point3D=keyFrame->mvLocalMapPoints[m].getPosition();
                    
                    point3d.x=point3D(0);
                    point3d.y=point3D(1);
                    point3d.z=point3D(2);
                    
                    Eigen::Vector3d normalized3d=*closeMeasurements[m];
                    point2d.x=normalized3d(0)/normalized3d(2);
                    point2d.y=normalized3d(1)/normalized3d(2);
                    
                    points3d.push_back(point3d);
                    points2d.push_back(point2d);
                    
                    //printf("3d %f %f %f %f %f\n",point3d.x,point3d.y,point3d.z,point2d.x,point2d.y);
                }
            }
            assert(points3d.size()>10);
            
            cv::Mat cameraMatrix=cv::Mat::eye(3,3,CV_64FC1);
            cv::Mat rVec,tVec,rMat;
            
            cv::solvePnPRansac(points3d,points2d,cameraMatrix,cv::Mat(),rVec,tVec,false,1000,0.008,0.99);
            cv::Rodrigues(rVec,rMat);
            
            Transform pose;
            for(int r1=0;r1<3;r1++){
                for(int r2=0;r2<3;r2++){
                    pose.rotation(r1,r2)=rMat.at<double>(r1,r2);
                }
            }
            
            tVec=-rMat.t()*tVec;
            assert(rMat.type()==CV_64FC1&&tVec.type()==CV_64FC1);
            pose.translation(0)=tVec.at<double>(0);
            pose.translation(1)=tVec.at<double>(1);
            pose.translation(2)=tVec.at<double>(2);
            
            
            double param_cam[6];
            ceres::RotationMatrixToAngleAxis(&pose.rotation(0),&param_cam[0]);
            Eigen::Vector3d translation=pose.rotation*pose.translation;
            param_cam[3]=-translation(0);
            param_cam[4]=-translation(1);
            param_cam[5]=-translation(2);
            
            vector<double> param_invDepth(points3d.size());
            ceres::Problem problem;
            ceres::LossFunction* loss_function = new ceres::HuberLoss(0.005);
            int nInliers=0;
            for (int i=0;i<points3d.size();i++) {
                cv::Mat point3dMat(3,1,CV_64FC1);
                point3dMat.at<double>(0)=points3d[i].x;
                point3dMat.at<double>(1)=points3d[i].y;
                point3dMat.at<double>(2)=points3d[i].z;
                
                
                cv::Mat proj=rMat*point3dMat+tVec;
                proj/=proj.at<double>(2);
                
                double diffx=proj.at<double>(0)-points2d[i].x;
                double diffy=proj.at<double>(1)-points2d[i].y;
                double error=std::sqrt(diffx*diffx+diffy*diffy);
                
                param_invDepth[i]=1.0/point3dMat.at<double>(2);
                point3dMat/=point3dMat.at<double>(2);
                
                if (error<0.005) {
                    ceres::CostFunction* cost_function=SnavelyReprojectionError::Create(point3dMat.at<double>(0),
                                                                                        point3dMat.at<double>(1),
                                                                                        (double)points2d[i].x,
                                                                                        (double)points2d[i].y);
                    problem.AddResidualBlock(cost_function,loss_function,param_cam,&param_invDepth[i]);
                    problem.SetParameterBlockConstant(&param_invDepth[i]);
                    nInliers++;
                }
            }
            
            ceres::Solver::Options options;
            options.max_num_iterations=5;
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout=false;
            ceres::Solver::Summary summary;
            ceres::Solve(options,&problem, &summary);
            
            
            ceres::AngleAxisToRotationMatrix(param_cam,&(keyFrame->mvFarFrames[i].pose.rotation(0)));
            keyFrame->mvFarFrames[i].pose.translation=-Eigen::Vector3d(&param_cam[3]);
            keyFrame->mvFarFrames[i].pose.translation=keyFrame->mvFarFrames[i].pose.rotation.transpose()
                                                     *keyFrame->mvFarFrames[i].pose.translation;
            keyFrame->mvFarFrames[i].pose.setEssentialMatrix();
        }
        return;
    }*/
    
    
    void LocalBundleAdjustment::refineKeyFrameConnection(KeyFrame* keyFrame){
        
        return;
        int num_keyframe=keyFrame->mConnectedKeyFrameMatches.size();
        int num_point=keyFrame->mvLocalMapPoints.size();
        
        if (num_keyframe==0||(keyFrame->nextKeyFramePtr!=NULL&&num_keyframe==1)) {
            return;
        }
        printf("%d %d\n",num_keyframe,num_point);
        std::vector<double> param_cam(6*num_keyframe);
        std::vector<double> param_invDepth(num_point);
        std::map<KeyFrame*,int> keyFrameIndex;
        
        int f=0;
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame->mConnectedKeyFramePoses.begin(),
            mend=keyFrame->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            
            if (mit->first==keyFrame->nextKeyFramePtr) {//next keyframe's pose is by bundle
                continue;
            }
            
            keyFrameIndex[mit->first]=f;
            ceres::RotationMatrixToAngleAxis(&(mit->second.rotation(0)),&param_cam[6*f]);
            Eigen::Vector3d translation=mit->second.rotation*mit->second.translation;
            param_cam[6*f+3]=-translation(0);
            param_cam[6*f+4]=-translation(1);
            param_cam[6*f+5]=-translation(2);
            f++;
        }
        
        std::vector<bool> depthFixed(num_point,true);
        
        for (int p=0;p<keyFrame->mvLocalMapPoints.size();p++) {
            param_invDepth[p]=keyFrame->mvLocalMapPoints[p].invdepth;
        }
        
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.005);
        
        for(map<KeyFrame*,std::vector<int> >::iterator mit=keyFrame->mConnectedKeyFrameMatches.begin(),
            mend=keyFrame->mConnectedKeyFrameMatches.end();
            mit!=mend;
            mit++){
            
            KeyFrame* connectedKeyFrame=mit->first;
            if (connectedKeyFrame==keyFrame->nextKeyFramePtr) {
                continue;
            }
            
            f=keyFrameIndex[connectedKeyFrame];
            std::vector<int> &matches=mit->second;
            for (int i=0;i<matches.size();i++) {
                
                if (matches[i]<0) {//not matched
                    continue;
                }
                
                if (!keyFrame->mvLocalMapPoints[i].isEstimated) {
                    assert(0);
                    std::vector<Eigen::Vector3d> points(2);
                    std::vector<Eigen::Matrix3d> rotations(2,Eigen::Matrix3d::Identity());
                    std::vector<Eigen::Vector3d> cameras(2);
                    
                    Transform transform=keyFrame->mConnectedKeyFramePoses[connectedKeyFrame];
                    
                    points[0]=keyFrame->mvLocalMapPoints[i].norm;
                    points[1]=keyFrame->mvLocalMapPoints[matches[i]].norm;
                    points[1]=transform.rotation.transpose()*points[1];
                    
                    cameras[0]=Eigen::Vector3d::Zero();
                    cameras[1]=transform.translation;
                    
                    //assert(keyFrame->mvLocalMapPoints[i].previousInvDepth==0.0);
                    Eigen::Vector3d point3D=multiviewTriangulationLinear(points,rotations,cameras);
                    keyFrame->mvLocalMapPoints[i].invdepth=1.0/point3D(2);
                    keyFrame->mvLocalMapPoints[i].isEstimated=true;
                    
                }
                assert(keyFrame->mvLocalMapPoints[i].isEstimated);
                
                Eigen::Vector3d keyFramePoint=keyFrame->mvLocalMapPoints[i].vec;
                Eigen::Vector3d framePoint=connectedKeyFrame->mvLocalMapPoints[matches[i]].vec;
                ceres::CostFunction* cost_function=SnavelyReprojectionError::Create(keyFramePoint(0),keyFramePoint(1),
                                                                                    framePoint(0),framePoint(1));
                problem.AddResidualBlock(cost_function,loss_function,&param_cam[6*f],&param_invDepth[i]);
            }
        }
        
        for (int i=0;i<num_point;i++) {
            if (keyFrame->mvLocalMapPoints[i].isEstimated&&keyFrame->mvLocalMapPoints[i].measurementCount>0) {
                problem.SetParameterBlockConstant(&param_invDepth[i]);
            }
        }
        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout=false;
        options.logging_type=ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options,&problem, &summary);
        
        
        for (int i=0;i<num_point;i++) {
            if (keyFrame->mvLocalMapPoints[i].isEstimated&&keyFrame->mvLocalMapPoints[i].measurementCount<=0) {
                keyFrame->mvLocalMapPoints[i].invdepth=param_invDepth[i];
                keyFrame->mvLocalMapPoints[i].updateNormalAndDepth();
            }
        }
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame->mConnectedKeyFramePoses.begin(),
            mend=keyFrame->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            
            if (mit->first==keyFrame->nextKeyFramePtr) {
                continue;
            }
            f=keyFrameIndex[mit->first];
            ceres::AngleAxisToRotationMatrix(&param_cam[6*f],&(mit->second.rotation(0)));
            mit->second.translation=-Eigen::Vector3d(&param_cam[6*f+3]);
            mit->second.translation=mit->second.rotation.transpose()*mit->second.translation;
            mit->second.setEssentialMatrix();
        }
        return;
    }
    
    int LocalBundleAdjustment::refineKeyFrameMatches(KeyFrame *keyFrame1,const KeyFrame *keyFrame2,Transform &transform,
                                                     vector<int>& matches12,vector<int>& matches21){
        int nMatches=0;
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(projErrorThres);
        
        double param_cam[6];
        
        ceres::RotationMatrixToAngleAxis(&(transform.rotation(0)),&param_cam[0]);
        Eigen::Vector3d translation=transform.rotation*transform.translation;
        param_cam[3]=-translation(0);
        param_cam[4]=-translation(1);
        param_cam[5]=-translation(2);
        
        vector<double> param_invDepth(matches12.size());
        std::vector<bool> depthFixed(matches12.size(),false);
        
        for (int i=0;i<matches12.size();i++) {
            
            if (matches12[i]<0) {
                continue;
            }
            
            if (!keyFrame1->mvLocalMapPoints[i].isEstimated) {
                std::vector<Eigen::Vector3d> points(2);
                std::vector<Eigen::Matrix3d> rotations(2,Eigen::Matrix3d::Identity());
                std::vector<Eigen::Vector3d> cameras(2);
                
                points[0]=keyFrame1->mvLocalMapPoints[i].norm;
                points[1]=keyFrame2->mvLocalMapPoints[matches12[i]].norm;
                points[1]=transform.rotation.transpose()*points[1];
                
                cameras[0]=Eigen::Vector3d::Zero();
                cameras[1]=transform.translation;
                
                Eigen::Vector3d point3D=multiviewTriangulationLinear(points,rotations,cameras);
                keyFrame1->mvLocalMapPoints[i].invdepth=1.0/point3D(2);
            }else{
                depthFixed[i]=true;
            }
            param_invDepth[i]=keyFrame1->mvLocalMapPoints[i].invdepth;
        }
        
        for (int i=0;i<matches12.size();i++) {
            if (matches12[i]<0) {
                continue;
            }else{
                nMatches++;
                Eigen::Vector3d keyFramePoint=keyFrame1->mvLocalMapPoints[i].vec;
                Eigen::Vector3d framePoint=keyFrame2->mvLocalMapPoints[matches12[i]].vec;
                ceres::CostFunction* cost_function=SnavelyReprojectionError::Create(keyFramePoint(0),keyFramePoint(1),
                                                                                    framePoint(0),framePoint(1));
                problem.AddResidualBlock(cost_function,loss_function,param_cam,&param_invDepth[i]);
            }
        }
        
        for (int i=0;i<matches12.size();i++){
            if(depthFixed[i]){
                problem.SetParameterBlockConstant(&param_invDepth[i]);
            }
        }
        
        ceres::Solver::Options options;
        options.max_num_iterations=BAIterations;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout=false;
        options.logging_type=ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options,&problem, &summary);
        //printf("%d %d before refine match %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches);
        for (int i=0;i<matches12.size();i++) {
            if (matches12[i]<0) {
                continue;
            }
            const Eigen::Vector3d point3D=keyFrame1->mvLocalMapPoints[i].getPosition();
            Eigen::Vector3d proj=transform.rotation*(point3D-transform.translation);
            proj/=proj(2);
            double error=(proj-keyFrame2->mvLocalMapPoints[matches12[i]].vec).norm();
            
            if (error>projErrorThres||param_invDepth[i]<0) {
                if(!matches21.empty()){
                    matches21[matches12[i]]=-1;
                }
                matches12[i]=-1;
                nMatches--;
            }else if(!depthFixed[i]){
                keyFrame1->mvLocalMapPoints[i].isEstimated=true;
                keyFrame1->mvLocalMapPoints[i].invdepth=param_invDepth[i];
                keyFrame1->mvLocalMapPoints[i].updateNormalAndDepth();
            }
        }
        
        ceres::AngleAxisToRotationMatrix(param_cam,&(transform.rotation(0)));
        transform.translation=-Eigen::Vector3d(&param_cam[3]);
        transform.translation=transform.rotation.transpose()*transform.translation;
        transform.setEssentialMatrix();
        return nMatches;
    }
    
    
    
    int PnPEstimator::estimate(KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                               const vector<int>& matches,Transform& pose){
        
        
        std::vector<int> indices;
        std::vector<cv::Point3f> points3d;
        std::vector<cv::Point2f> points2d;
        
        indices.reserve(matches.size());
        points3d.reserve(matches.size());
        points2d.reserve(matches.size());
        
        //std::cout<<"size "<<matches.size()<<std::endl;
        assert(matches.size()>20);
        for(int i=0;i<matches.size();i++){
            if(matches[i]>=0&&keyFrame1->mvLocalMapPoints[i].isEstimated==true){
                
                cv::Point3f point3d;
                cv::Point2f point2d;
                Eigen::Vector3d point3D=keyFrame1->mvLocalMapPoints[i].getPosition();
                
                point3d.x=point3D(0);
                point3d.y=point3D(1);
                point3d.z=point3D(2);
                
                Eigen::Vector3d normalized3d=keyFrame2->mvLocalMapPoints[matches[i]].norm;
                point2d.x=normalized3d(0)/normalized3d(2);
                point2d.y=normalized3d(1)/normalized3d(2);
                
                points3d.push_back(point3d);
                points2d.push_back(point2d);
                indices.push_back(i);
            }
        }
        
        points3d.shrink_to_fit();
        points2d.shrink_to_fit();
        indices.shrink_to_fit();
        int nInliers=indices.size();
        
        cv::Mat cameraMatrix=cv::Mat::eye(3,3,CV_64FC1);
        cv::Mat rVec,tVec,rMat;
        
        cv::solvePnP(points3d,points2d,cameraMatrix,cv::Mat(),rVec,tVec);
        cv::Rodrigues(rVec,rMat);
        
        for(int r1=0;r1<3;r1++){
            for(int r2=0;r2<3;r2++){
                pose.rotation(r1,r2)=rMat.at<double>(r1,r2);
            }
        }
        
        tVec=-rMat.t()*tVec;
        assert(rMat.type()==CV_64FC1&&tVec.type()==CV_64FC1);
        pose.translation(0)=tVec.at<double>(0);
        pose.translation(1)=tVec.at<double>(1);
        pose.translation(2)=tVec.at<double>(2);
        
        
        return nInliers;
    }
    
    int PnPEstimator::estimate(KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                               vector<int>& matches12,vector<int>& matches21,Transform& pose){
        
        
        std::vector<int> indices;
        std::vector<cv::Point3f> points3d;
        std::vector<cv::Point2f> points2d;
        
        indices.reserve(matches12.size());
        points3d.reserve(matches12.size());
        points2d.reserve(matches12.size());
        
        //std::cout<<"size "<<matches.size()<<std::endl;
        for(int i=0;i<matches12.size();i++){
            if(matches12[i]>=0&&keyFrame1->mvLocalMapPoints[i].isEstimated==true){
                
                cv::Point3f point3d;
                cv::Point2f point2d;
                Eigen::Vector3d point3D=keyFrame1->mvLocalMapPoints[i].getPosition();
                
                point3d.x=point3D(0);
                point3d.y=point3D(1);
                point3d.z=point3D(2);
                
                Eigen::Vector3d normalized3d=keyFrame2->mvLocalMapPoints[matches12[i]].norm;
                point2d.x=normalized3d(0)/normalized3d(2);
                point2d.y=normalized3d(1)/normalized3d(2);
                
                points3d.push_back(point3d);
                points2d.push_back(point2d);
                indices.push_back(i);
            }
        }
        
        points3d.shrink_to_fit();
        points2d.shrink_to_fit();
        indices.shrink_to_fit();
        int nInliers=indices.size();
        
        cv::Mat cameraMatrix=cv::Mat::eye(3,3,CV_64FC1);
        cv::Mat rVec,tVec,rMat;
        
        std::vector<uchar> status(points3d.size());
        cv::solvePnPRansac(points3d,points2d,cameraMatrix,cv::Mat(),rVec,tVec,false,1000,threshold,prob);
        cv::Rodrigues(rVec,rMat);
        
        for(int r1=0;r1<3;r1++){
            for(int r2=0;r2<3;r2++){
                pose.rotation(r1,r2)=rMat.at<double>(r1,r2);
            }
        }
        
        tVec=-rMat.t()*tVec;
        assert(rMat.type()==CV_64FC1&&tVec.type()==CV_64FC1);
        pose.translation(0)=tVec.at<double>(0);
        pose.translation(1)=tVec.at<double>(1);
        pose.translation(2)=tVec.at<double>(2);
        
        
        for (int i=0;i<matches12.size();i++) {
            if(matches12[i]>=0&&keyFrame1->mvLocalMapPoints[i].isEstimated==true){
                
                Eigen::Vector3d proj=keyFrame1->mvLocalMapPoints[i].getPosition();
                proj=pose.rotation*(proj-pose.translation);
                proj/=proj(2);
                
                Eigen::Vector3d vec=keyFrame2->mvLocalMapPoints[matches12[i]].vec;
                Eigen::Vector3d error=proj-vec;
                
                if (error.norm()>threshold) {
                    matches21[matches12[i]]=-1;
                    matches12[i]=-1;
                    nInliers--;
                }
            }
        }
        return nInliers;
    }
}




