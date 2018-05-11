//
//  Geometry.cpp
//  GSLAM
//
//  Created by ctang on 9/17/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include <stdio.h>
#include "Geometry.h"
#include "ceres/ceres.h"

Eigen::Vector3d multiviewTriangulationLinear(const std::vector<Eigen::Vector3d> &points,
                                             const std::vector<Eigen::Matrix3d> &rotations,
                                             const std::vector<Eigen::Vector3d> &cameras){
    Eigen::Vector3d point3D;
    int ncamera=points.size();
    Eigen::MatrixX4d A=Eigen::MatrixX4d::Zero(3*ncamera,4);
    Eigen::MatrixX4d pLastRow=Eigen::MatrixX4d::Zero(ncamera,4);
    for(int i=0;i<ncamera;i++){
        Eigen::Matrix3d crossMat;
        Eigen::MatrixX4d P=Eigen::MatrixX4d(3,4);
        setCrossMatrix(crossMat,points[i]);
        P.block<3,3>(0,0)=rotations[i];
        P.col(3)=-rotations[i]*cameras[i];
        A.block<3,4>(3*i,0)=crossMat*P;
        pLastRow.row(i)=P.row(2);
    }
    Eigen::Matrix4d AtA=A.transpose()*A;
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(AtA,Eigen::ComputeFullV);
    Eigen::Vector4d solution=svd.matrixV().col(3);
    
    point3D(0)=solution(0)/solution(3);
    point3D(1)=solution(1)/solution(3);
    point3D(2)=solution(2)/solution(3);
    
    Eigen::VectorXd product=pLastRow*solution;
    int positiveCount=0;
    for(int i=0;i<ncamera;i++){
        positiveCount+=(product(i)>0);
    }
    
    if(positiveCount<ncamera/2){
        solution=-solution;
    }
    
    point3D(0)=solution(0)/solution(3);
    point3D(1)=solution(1)/solution(3);
    point3D(2)=solution(2)/solution(3);
    
    return point3D;
}


/*void    multiviewTriangulationNonliear(double& depth,
                                       const std::vector<Eigen::Vector3d>& points,
                                       const std::vector<Eigen::Matrix3d>& rotations,
                                       const std::vector<Eigen::Vector3d>& cameras){
    
    
}*/




double  multiviewTriangulationDepth(const std::vector<Eigen::Vector3d> &points,
                                    const std::vector<Eigen::Matrix3d> &rotations,
                                    const std::vector<Eigen::Vector3d> &cameras){
    
    
    int ncameras=points.size()-1;
    Eigen::VectorXd  A=Eigen::VectorXd(3*ncameras);
    Eigen::VectorXd  B=Eigen::VectorXd(3*ncameras);
    
    for (int i=1;i<=ncameras;i++) {
        Eigen::Matrix3d crossMat;
        Eigen::Vector3d rotated=rotations[i].transpose()*points[i];
        setCrossMatrix(crossMat,rotated);
        rotated=crossMat*points[0];
        A.block<3,1>(3*(i-1),0)=rotated;
        B.block<3,1>(3*(i-1),0)=crossMat*cameras[i];
    }
    
    double a=A.dot(A);
    double b=A.dot(B);
    return std::abs(b/a);
}


/*void    multiviewTriangulationNonliear(double& depth,
                                       const std::vector<Eigen::Vector3d>& points,
                                       const std::vector<Eigen::Matrix3d>& rotations,
                                       const std::vector<Eigen::Vector3d>& cameras){
    
    
}*/



inline Eigen::Matrix3d AngleAxis2RotationMatrix(const double costheta,
                                                const double sintheta,
                                                const double wx,
                                                const double wy,
                                                const double wz){
    
    Eigen::Matrix3d R;
    R(0,0) =     costheta   + wx*wx*(1 -    costheta);
    R(1,0) =  wz*sintheta   + wx*wy*(1 -    costheta);
    R(2,0) = -wy*sintheta   + wx*wz*(1 -    costheta);
    R(0,1) =  wx*wy*(1 - costheta)     - wz*sintheta;
    R(1,1) =     costheta   + wy*wy*(1 -    costheta);
    R(2,1) =  wx*sintheta   + wy*wz*(1 -    costheta);
    R(0,2) =  wy*sintheta   + wx*wz*(1 -    costheta);
    R(1,2) = -wx*sintheta   + wy*wz*(1 -    costheta);
    R(2,2) =     costheta   + wz*wz*(1 -    costheta);
    return R;
}

Eigen::Matrix3d ComputeAxisRotation(const Eigen::Vector3d &v1,const Eigen::Vector3d &v2){
    double cos12=v1.dot(v2)/(v1.norm()*v2.norm());
    double sin12=sqrt(1-cos12*cos12);
    Eigen::Vector3d axis =v1.cross(v2);
    axis.normalize();
    Eigen::Matrix3d R=AngleAxis2RotationMatrix(cos12,sin12,axis(0),axis(1),axis(2));
    return R;
}


