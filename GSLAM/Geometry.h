//
//  Geometry.hpp
//  GSLAM
//
//  Created by ctang on 9/17/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#ifndef Geometry_h
#define Geometry_h

#include "eigen3/Eigen/Dense"
#include <vector>

inline void setCrossMatrix(Eigen::Matrix3d &crossMat,const Eigen::Vector3d& vector){
    crossMat(0,0)=crossMat(1,1)=crossMat(2,2)=0.0;
    crossMat(0,1)=  vector(2);
    crossMat(0,2)= -vector(1);
    crossMat(1,0)= -vector(2);
    crossMat(1,2)=  vector(0);
    crossMat(2,0)=  vector(1);
    crossMat(2,1)= -vector(0);
}


Eigen::Vector3d multiviewTriangulationLinear(const std::vector<Eigen::Vector3d> &points,
                                             const std::vector<Eigen::Matrix3d> &rotations,
                                             const std::vector<Eigen::Vector3d> &cameras);

double          multiviewTriangulationDepth(const std::vector<Eigen::Vector3d> &points,
                                            const std::vector<Eigen::Matrix3d> &rotations,
                                            const std::vector<Eigen::Vector3d> &cameras);

void            multiviewTriangulationNonlinear(double& depth,
                                                const std::vector<Eigen::Vector3d> &points,
                                                const std::vector<Eigen::Matrix3d> &rotations,
                                                const std::vector<Eigen::Vector3d> &cameras);


inline Eigen::Matrix3d AngleAxis2RotationMatrix(const double costheta,
                                                const double sintheta,
                                                const double wx,
                                                const double wy,
                                                const double wz);

Eigen::Matrix3d ComputeAxisRotation(const Eigen::Vector3d &v1,const Eigen::Vector3d &v2);

#endif /* Geometry_h */
