//
//  GlobalRotationEstimation.cpp
//  GSLAM
//
//  Created by ctang on 10/2/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "GlobalReconstruction.h"
#include "L1Solver.h"
#include "opencv2/core/core.hpp"
#include <sys/time.h>


namespace GSLAM{
    
    
    void GlobalRotationEstimation::solve(const std::vector<RotationConstraint>& constraints,
                                         std::vector<Eigen::Matrix3d>& results){
        
        static const double kConvergenceThreshold = 1e-2;
        
        int variableCount=results.size();
        int constraintCount=constraints.size();
        int rowCount=2*constraintCount;
        
        int columnCount=variableCount+constraintCount;
        int singleVariableConstraintCount=0;
        
        for(int i=0;i<constraintCount;i++){
            singleVariableConstraintCount+=(constraints[i].variableIndex1==-1);
             assert(constraints[i].variableIndex2!=-1);
        }
        int elementCount=3*rowCount-2*singleVariableConstraintCount;
        
        
        
        if (objective==NULL) {
            
            objective=(double*)malloc(columnCount*sizeof(double));
            rowStart=(int*)malloc((rowCount+1)*sizeof(int));
            column=(int*)malloc(elementCount*sizeof(int));
            
            rowUpper=(double*)malloc(rowCount*sizeof(double));
            rowLower=(double*)malloc(rowCount*sizeof(double));
            
            columnUpper=(double*)malloc(columnCount*sizeof(double));
            columnLower=(double*)malloc(columnCount*sizeof(double));
            
            elementByRow=(double*)malloc(elementCount*sizeof(double));
            
        }else{
            
            objective=(double*)realloc(objective,columnCount*sizeof(double));
            rowStart=(int*)realloc(rowStart,(rowCount+1)*sizeof(int));
            column=(int*)realloc(column,elementCount*sizeof(int));
            
            rowUpper=(double*)realloc(rowUpper,rowCount*sizeof(double));
            rowLower=(double*)realloc(rowLower,rowCount*sizeof(double));
            
            columnUpper=(double*)realloc(columnUpper,columnCount*sizeof(double));
            columnLower=(double*)realloc(columnLower,columnCount*sizeof(double));
            elementByRow=(double*)realloc(elementByRow,elementCount*sizeof(double));
        }
        
        for (int i = 0; i < variableCount; ++i){
            columnLower[i]=-COIN_DBL_MAX;
            columnUpper[i]=COIN_DBL_MAX;
        }
        
        for(int i=variableCount;i<columnCount;i++){
            columnLower[i]=0.0;
            columnUpper[i]=COIN_DBL_MAX;
        }
        
        int elementIndex=0;
        int *rowStartPtr=&rowStart[0];
        int constraintOffset=variableCount;
        
        for (int c = 0; c < constraintCount; c++){
            
            rowStartPtr[2*c]=elementIndex;
            if(constraints[c].variableIndex1!=-1){
                column[elementIndex]=constraints[c].variableIndex1;
                elementByRow[elementIndex++]=-1;
            }
            
            if(constraints[c].variableIndex2!=-1){
                column[elementIndex]=constraints[c].variableIndex2;
                elementByRow[elementIndex++]=1;
            }
            
            column[elementIndex]=constraintOffset+c;
            elementByRow[elementIndex++]=-1;
            
            rowStartPtr[2*c+1]=elementIndex;
            
            if(constraints[c].variableIndex1!=-1){
                column[elementIndex]=constraints[c].variableIndex1;
                elementByRow[elementIndex++]=-1;
            }
            
            if(constraints[c].variableIndex2!=-1){
                column[elementIndex]=constraints[c].variableIndex2;
                elementByRow[elementIndex++]=1;
            }
            
            column[elementIndex]=constraintOffset+c;
            elementByRow[elementIndex++]=1;
        }
        
        rowStart[rowCount]=elementIndex;
        assert(elementIndex==elementCount);
        
        memset(objective,0,variableCount*sizeof(double));
        for(int i=variableCount;i<columnCount;i++){
            objective[i]=constraints[i-variableCount].weight;
        }
        
        ClpSimplex model;
        CoinPackedMatrix byRow(false,columnCount,rowCount,elementCount,
                               elementByRow,column,rowStart,NULL);
        
        
        std::vector<Eigen::Vector3d> rotationErrors(constraintCount);
        std::vector<Eigen::Vector3d> rotationUpdates(variableCount);
        
        double rotationError=0.0;
        for (int iter=0;iter<maxOuterIterations;iter++) {
            //evaluate error
            for (int i=0;i<constraintCount;i++) {
                assert(constraints[i].variableIndex2!=-1);
                if (constraints[i].variableIndex1!=-1) {
                    /*std::cout<<constraints[i].variableIndex1<<' '<<constraints[i].variableIndex2<<std::endl;
                    std::cout<<constraints[i].rotation12<<std::endl;
                    std::cout<<results[constraints[i].variableIndex1]<<std::endl;
                    std::cout<<results[constraints[i].variableIndex2]<<std::endl;*/
                    rotationErrors[i]=ComputeRelativeRotationError(constraints[i].rotation12,
                                                                   results[constraints[i].variableIndex1],
                                                                   results[constraints[i].variableIndex2]);
                }else{
                    rotationErrors[i]=ComputeRelativeRotationError(constraints[i].rotation12,
                                                                   constraints[i].rotation1,
                                                                   results[constraints[i].variableIndex2]);

                }
                rotationError+=rotationErrors[i].norm();
            }
            rotationError/=constraintCount;
            if (rotationError<kConvergenceThreshold) {
                
                break;
            }
            
            for (int r=0;r<3;r++) {
                
                for (int i=0;i<constraintCount;i++) {
                    rowUpper[2*i]   =rotationErrors[i](r);
                    rowLower[2*i]   =-COIN_DBL_MAX;
                    rowUpper[2*i+1] =COIN_DBL_MAX;
                    rowLower[2*i+1] =rotationErrors[i](r);
                }
                
                model.setLogLevel(0);
                model.loadProblem(byRow,columnLower,columnUpper,objective,rowLower,rowUpper);
                model.setMaximumIterations(maxInnerIterations);
                model.dual();
                
                
                const double *solutionPtr=model.primalColumnSolution();
                for (int i=0;i<variableCount;i++) {
                    rotationUpdates[i](r)=solutionPtr[i];
                }
            }
            
            for (int i=0;i<variableCount;i++) {
                ApplyRotation(rotationUpdates[i],results[i]);
            }
            maxInnerIterations*=2;
        }
    }

    
    void GlobalRotationEstimation::test(){
        
        int num_rotations=1000;
        std::vector<Eigen::Matrix3d> globalRotations(num_rotations);
        std::vector<RotationConstraint> constraints;
        cv::RNG rng(-1);
        for (int i=0;i<globalRotations.size();i++) {
            double angles[3]={rng.uniform(-10.0,10.0),rng.uniform(-10.0,10.0),rng.uniform(-10.0,10.0)};
            for (int r=0;r<3;r++) {
                angles[r]*=(CV_PI/180.0);
            }
            ceres::AngleAxisToRotationMatrix(angles,ceres::ColumnMajorAdapter3x3(globalRotations[i].data()));
        }
        
        for (int i=0;i<num_rotations;i++) {
            RotationConstraint constraint;
            constraint.variableIndex1=i;
            for (int j=i+1;j<std::min(i+5,num_rotations);j++) {
                constraint.variableIndex2=j;
                constraint.rotation12=globalRotations[j]*globalRotations[i].transpose();
                constraint.weight=1.0;
                constraints.push_back(constraint);
            }
            
            if (i!=0) {
                continue;
            }
            constraint.variableIndex1=-1;
            constraint.variableIndex2=i;
            constraint.rotation1=Eigen::Matrix3d::Identity();
            constraint.rotation12=globalRotations[i];
            constraint.weight=1.0;
            constraints.push_back(constraint);
            
            if (i<num_rotations&&i>num_rotations-100) {
                constraint.variableIndex1=0;
                constraint.variableIndex2=i;
                constraint.rotation12=globalRotations[i]*globalRotations[0].transpose();
                constraint.weight=1.0;
                constraints.push_back(constraint);
            }
        }
        
        for (int i=0;i<constraints.size();i++) {
            
            double angles[3]={rng.uniform(-0.5,0.5),rng.uniform(-0.5,0.5),rng.uniform(-0.5,0.5)};
            
            for (int r=0;r<3;r++) {
                angles[r]*=(CV_PI/180.0);
            }
            
            Eigen::Matrix3d noisyRotation;
            ceres::AngleAxisToRotationMatrix(angles,ceres::ColumnMajorAdapter3x3(noisyRotation.data()));
            constraints[i].rotation12=noisyRotation*constraints[i].rotation12;
        }
        std::vector<Eigen::Matrix3d> results(6,Eigen::Matrix3d::Identity());
        int i=0;
        std::vector<RotationConstraint> incrementalConstraints;
        
        for (int f=5;f<num_rotations;f++) {
            for (;i<constraints.size();i++) {
                if (constraints[i].variableIndex2<=f) {
                    incrementalConstraints.push_back(constraints[i]);
                }else{
                    break;
                }
            }
            if (f==5) {
                this->solve(incrementalConstraints,results);
            }else{
                std::vector<double> wx,wy,wz;
                for (int j=incrementalConstraints.size()-1;j>0;j--) {
                    if (incrementalConstraints[j].variableIndex2==f) {
                        const Eigen::Matrix3d rotationf=incrementalConstraints[j].rotation12*results[incrementalConstraints[j].variableIndex1];
                        double angles[3];
                        ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(rotationf.data()),angles);
                        wx.push_back(angles[0]);
                        wy.push_back(angles[1]);
                        wz.push_back(angles[2]);
                    }
                }
                std::sort(wx.begin(),wx.end());
                std::sort(wy.begin(),wy.end());
                std::sort(wz.begin(),wz.end());
                
                const double angles[3]={wx[wx.size()/2],wy[wy.size()/2],wz[wz.size()/2]};
                Eigen::Matrix3d rotationf;
                ceres::AngleAxisToRotationMatrix(angles,ceres::ColumnMajorAdapter3x3(rotationf.data()));
                results.push_back(rotationf);
                this->solve(incrementalConstraints,results);
            }
        }

        std::vector<Eigen::Matrix3d> results2(num_rotations,Eigen::Matrix3d::Identity());
        this->solve(constraints,results2);
        
        double error0=0.0,error1=0.0,error2=0;
        
        for (int i=1;i<num_rotations;i++) {
            std::cout<<"****"<<i<<"****"<<std::endl;
            std::cout<<results[i]*results[0].transpose()<<std::endl;
            std::cout<<"**********"<<std::endl;
            std::cout<<globalRotations[i]*globalRotations[0].transpose()<<std::endl;
            std::cout<<"**********"<<std::endl;
            std::cout<<results2[i]*results2[0].transpose()<<std::endl;
            std::cout<<"**********"<<std::endl;
            
            const Eigen::Matrix3d rotationError0=globalRotations[0]*globalRotations[i].transpose()*results[i]*results[0].transpose();
            const Eigen::Matrix3d rotationError1=globalRotations[0]*globalRotations[i].transpose()*results2[i]*results2[0].transpose();
            const Eigen::Matrix3d rotationError2=globalRotations[i].transpose()*results2[i];
            
            double q[4];
            ceres::RotationMatrixToQuaternion(ceres::ColumnMajorAdapter3x3(rotationError0.data()),q);
            error0+=std::abs(q[3]);
            
            ceres::RotationMatrixToQuaternion(ceres::ColumnMajorAdapter3x3(rotationError1.data()),q);
            error1+=std::abs(q[3]);
            
            ceres::RotationMatrixToQuaternion(ceres::ColumnMajorAdapter3x3(rotationError2.data()),q);
            error2+=std::abs(q[3]);
            
            
        }
        
        printf("%f %f %f\n",error0/num_rotations,error1/num_rotations,error2/num_rotations);
    }
}
