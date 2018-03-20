//
//  RelativeMotion.hpp
//  GSLAM
//
//  Created by Chaos on 2/17/17.
//  Copyright © 2017 ctang. All rights reserved.
//

#ifndef RelativeMotion_h
#define RelativeMotion_h

#include "ceres/rotation.h"
#include "ceres/ceres.h"

struct RotationTranslation{
    
    RotationTranslation(const double _x1,const double _y1,const double _z1,
                        const double _x2,const double _y2,const double _z2):
    x1(_x1),y1(_y1),z1(_z1),x2(_x2),y2(_y2),z2(_z2){
    }
    
    template <typename T>
    bool operator()(const T* const camera,
                    T* residuals) const {
        
        T rotated1[3];
        T p1[3]={(T)x1,(T)y1,(T)z1};
        T p2[3]={(T)x2,(T)y2,(T)z2};
        ceres::AngleAxisRotatePoint(camera,p1,rotated1);
        T norm=camera[3]*camera[3]+camera[4]*camera[4]+camera[5]*camera[5];
        
        if(sqrt(norm)>=(T)std::numeric_limits<double>::min()){//translation is not zero
            
            T axis[3];
            ceres::CrossProduct(rotated1,&camera[3],axis);
            T D=-(axis[0]*rotated1[0]+axis[1]*rotated1[1]+axis[2]*rotated1[2]);
            
            norm=axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2];
            T t=(axis[0]*p2[0]+axis[1]*p2[1]+axis[2]*p2[2]+D)/norm;
            
            rotated1[0]=p2[0]+axis[0]*t;
            rotated1[1]=p2[1]+axis[1]*t;
            rotated1[2]=p2[2]+axis[2]*t;
        }
        
        norm=sqrt(rotated1[0]*rotated1[0]+rotated1[1]*rotated1[1]+rotated1[2]*rotated1[2]);
        residuals[0]=rotated1[0]/norm-p2[0];
        residuals[1]=rotated1[1]/norm-p2[1];
        residuals[2]=rotated1[2]/norm-p2[2];
        return true;
    }
    
    static ceres::CostFunction* Create(const double x1,
                                       const double y1,
                                       const double z1,
                                       const double x2,
                                       const double y2,
                                       const double z2) {
        
        return (new ceres::AutoDiffCostFunction<RotationTranslation,3,6>(new RotationTranslation(x1,y1,z1,x2,y2,z2)));
        
    };
    
    double x1;
    double y1;
    double z1;
    double x2;
    double y2;
    double z2;
};



struct Rotation{
    
    Rotation(const double _x1,const double _y1,const double _z1,
             const double _x2,const double _y2,const double _z2):
    x1(_x1),y1(_y1),z1(_z1),x2(_x2),y2(_y2),z2(_z2){
    }
    
    template <typename T>
    bool operator()(const T* const camera,
                    T* residuals) const {
        
        T rotated1[3];
        T p1[3]={(T)x1,(T)y1,(T)z1};
        T p2[3]={(T)x2,(T)y2,(T)z2};
        ceres::AngleAxisRotatePoint(camera,p1,rotated1);
        residuals[0]=rotated1[0]-p2[0];
        residuals[1]=rotated1[1]-p2[1];
        residuals[2]=rotated1[2]-p2[2];
        return true;
    }
    
    static ceres::CostFunction* Create(const double x1,
                                       const double y1,
                                       const double z1,
                                       const double x2,
                                       const double y2,
                                       const double z2) {
        
        return (new ceres::AutoDiffCostFunction<Rotation,3,3>(new Rotation(x1,y1,z1,x2,y2,z2)));
        
    };
    
    double x1;
    double y1;
    double z1;
    double x2;
    double y2;
    double z2;
};

Eigen::Vector3d estimateRelativeTranslation(const std::vector<Eigen::Vector3d> &pts1,
                                            const std::vector<Eigen::Vector3d> &pts2){
    
    int num_point=pts1.size();
    std::vector<Eigen::Vector3d> norms(num_point);
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
    return curResult;
}


void preSolve(const std::vector<Eigen::Vector3d> &pts1,
              const std::vector<Eigen::Vector3d> &pts2,
              const Eigen::Matrix3d& rotation,
              Eigen::Vector3d& translation){
    
    std::vector<Eigen::Vector3d> _pts1(pts1.size()),_pts2(pts2.size());
    for(int p=0;p<pts1.size();p++){
        _pts1[p]=rotation*pts1[p];
        _pts2[p]=pts2[p];
    }
    translation=estimateRelativeTranslation(_pts1,_pts2);
    
    
    /*double motion[3]={0};
    ceres::RotationMatrixToAngleAxis(rotation.data(),&motion[0]);
    ceres::Problem problem;
    for(int i=0;i<pts1.size();i++){
        ceres::CostFunction* cost_function=Rotation::Create(pts1[i](0),
                                                            pts1[i](1),
                                                            pts1[i](2),
                                                            pts2[i](0),
                                                            pts2[i](1),
                                                            pts2[i](2));
        problem.AddResidualBlock(cost_function,NULL,motion);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout=false;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem, &summary);
    ceres::AngleAxisToRotationMatrix(motion,rotation.data());
    
    std::vector<Eigen::Vector3d> _pts1(pts1.size()),_pts2(pts2.size());
    
    for(int p=0;p<pts1.size();p++){
        _pts1[p]=rotation*pts1[p];
        _pts2[p]=pts2[p];
    }
    translation=estimateRelativeTranslation(_pts1,_pts2);*/
}

void estimateRotationTranslation(const double lossThreshold,
                                 const std::vector<Eigen::Vector3d> &pts1,
                                 const std::vector<Eigen::Vector3d> &pts2,
                                 Eigen::Matrix3d& rotation,
                                 Eigen::Vector3d& translation){
    
    
    assert(pts1.size()==pts2.size());
    preSolve(pts1,pts2,rotation,translation);
    
    double motion[6]={0};
    ceres::RotationMatrixToAngleAxis(rotation.data(),&motion[0]);
    motion[3]=translation(0);
    motion[4]=translation(1);
    motion[5]=translation(2);
    
    
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(lossThreshold);
    
    for(int i=0;i<pts1.size();i++){
        ceres::CostFunction* cost_function=RotationTranslation::Create(pts1[i](0),
                                                                       pts1[i](1),
                                                                       pts1[i](2),
                                                                       pts2[i](0),
                                                                       pts2[i](1),
                                                                       pts2[i](2));
        problem.AddResidualBlock(cost_function,NULL,motion);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout=false;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem, &summary);
    
    ceres::AngleAxisToRotationMatrix(motion,rotation.data());
    translation(0)=motion[3];
    translation(1)=motion[4];
    translation(2)=motion[5];
}

void estimateRotationTranslation(const double lossThreshold,
                                 const std::vector<Eigen::Vector3d> &pts1,
                                 const std::vector<Eigen::Vector3d> &pts2,
                                 std::vector<double>&   ratios,
                                 Eigen::Matrix3d& rotation,
                                 Eigen::Vector3d& translation){
    
    
    assert(pts1.size()==pts2.size());
    preSolve(pts1,pts2,rotation,translation);
    
    double motion[6]={0};
    ceres::RotationMatrixToAngleAxis(rotation.data(),&motion[0]);
    motion[3]=translation(0);
    motion[4]=translation(1);
    motion[5]=translation(2);
    
    
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(lossThreshold);
    
    for(int i=0;i<pts1.size();i++){
        ceres::CostFunction* cost_function=RotationTranslation::Create(pts1[i](0),
                                                                       pts1[i](1),
                                                                       pts1[i](2),
                                                                       pts2[i](0),
                                                                       pts2[i](1),
                                                                       pts2[i](2));
        problem.AddResidualBlock(cost_function,NULL,motion);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout=false;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem, &summary);
    
    
    ceres::AngleAxisToRotationMatrix(motion,rotation.data());
    translation(0)=motion[3];
    translation(1)=motion[4];
    translation(2)=motion[5];
}


struct Transform{
    
    Transform(const double _x1,const double _y1,const double _z1,
              const double _x2,const double _y2,const double _z2):
    x1(_x1),y1(_y1),z1(_z1),x2(_x2),y2(_y2),z2(_z2){
    }
    
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const radius,
                    T* residuals) const {
        
        T p[3];
        
        T point[3]={(T)x1,(T)y1,(T)z1};
        T norm=sqrt(camera[3]*camera[3]+camera[4]*camera[4]+camera[5]*camera[5]);
        ceres::AngleAxisRotatePoint(camera,point,p);
        p[0]+=(*radius)*(camera[3]-p[0]);
        p[1]+=(*radius)*(camera[4]-p[1]);
        p[2]+=(*radius)*(camera[5]-p[2]);
        
        norm=sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        residuals[0]=p[0]/norm-(T)x2;
        residuals[1]=p[1]/norm-(T)y2;
        residuals[2]=p[2]/norm-(T)z2;
        return true;
    }
    
    static ceres::CostFunction* Create(const double x1,
                                       const double y1,
                                       const double z1,
                                       const double x2,
                                       const double y2,
                                       const double z2) {
        
        return (new ceres::AutoDiffCostFunction<Transform,3,6,1>(new Transform(x1,y1,z1,x2,y2,z2)));
        
    };
    
    double x1;
    double y1;
    double z1;
    double x2;
    double y2;
    double z2;
};

void estimateRotationTranslation2(const double lossThreshold,
                                 const std::vector<Eigen::Vector3d> &pts1,
                                 const std::vector<Eigen::Vector3d> &pts2,
                                 Eigen::Matrix3d& rotation,
                                 Eigen::Vector3d& translation){
    
    double motion[6]={0};
    ceres::RotationMatrixToAngleAxis(rotation.data(),&motion[0]);
    motion[3]=translation(0);
    motion[4]=translation(1);
    motion[5]=translation(2);
    
    ceres::Problem problem;
    ceres::LossFunction* loss_function = NULL;
    std::vector<double> ratios(pts1.size(),0.0);
    
    
    for(int i=0;i<pts1.size();i++){
        ceres::CostFunction* cost_function=Transform::Create(pts1[i](0),pts1[i](1),pts1[i](2),
                                                             pts2[i](0),pts2[i](1),pts2[i](2));
        problem.AddResidualBlock(cost_function,loss_function,motion,&ratios[i]);
        problem.SetParameterLowerBound(&ratios[i],0,0.0);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout=false;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem, &summary);
    
    ceres::AngleAxisToRotationMatrix(motion,rotation.data());
    translation(0)=motion[3];
    translation(1)=motion[4];
    translation(2)=motion[5];
}

#endif /* RelativeMotion_h */
