//
//  GlobalTranslationEstimation.cpp
//  GSLAM
//
//  Created by ctang on 10/3/16.
//  Copyright © 2016 ctang. All rights reserved.
//
#include "GlobalReconstruction.h"
#include "L1Solver.h"
#include "opencv2/core/core.hpp"

namespace GSLAM{
    
    void GlobalTranslationEstimation::solve(const std::vector<TranslationConstraint> &constraints,std::vector<Eigen::Vector3d>& results){
        
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
            assert(constraints[c].variableIndex2!=-1);
            column[elementIndex]=constraints[c].variableIndex2;
            elementByRow[elementIndex++]=1;
            
            column[elementIndex]=constraintOffset+c;
            elementByRow[elementIndex++]=-1;
            
            rowStartPtr[2*c+1]=elementIndex;
            
            if(constraints[c].variableIndex1!=-1){
                column[elementIndex]=constraints[c].variableIndex1;
                elementByRow[elementIndex++]=-1;
            }
            
            assert(constraints[c].variableIndex2!=-1);
            column[elementIndex]=constraints[c].variableIndex2;
            elementByRow[elementIndex++]=1;
            
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
        //model.setLogLevel(0);
        //model.setMaximumIterations(this->maxIterations);
        
        for (int c=0;c<3;c++) {
            
            for (int i = 0; i <constraintCount; ++i){
                
                rowUpper[2*i]=constraints[i].translation12(c);
                rowLower[2*i]=-COIN_DBL_MAX;
                rowUpper[2*i+1]=COIN_DBL_MAX;
                rowLower[2*i+1]=constraints[i].translation12(c);
                
                if (constraints[i].variableIndex1==-1) {
                    rowUpper[2*i]+=constraints[i].translation1(c)-results[constraints[i].variableIndex2](c);
                    rowLower[2*i+1]+=constraints[i].translation1(c)-results[constraints[i].variableIndex2](c);
                }else{
                    rowUpper[2*i]+=results[constraints[i].variableIndex1](c)-results[constraints[i].variableIndex2](c);
                    rowLower[2*i+1]+=results[constraints[i].variableIndex1](c)-results[constraints[i].variableIndex2](c);
                }
            }
            
            model.loadProblem(byRow,columnLower,columnUpper,objective,rowLower,rowUpper);
            model.primal();
            const double *solutionPtr=model.primalColumnSolution();
            for (int i=0;i<variableCount;i++) {
                results[i](c)=solutionPtr[i];
            }
        }
    }
    
    
    void GlobalTranslationEstimation::solveComplex(const std::vector<TranslationConstraint> &constraints,std::vector<Eigen::Vector3d>& results){
        
    }
    
    void GlobalTranslationEstimation::test(){
        
        int num_translation=500;
        std::vector<Eigen::Vector3d> globalTranslations(num_translation);
        std::vector<Eigen::Matrix3d> globalRotations(num_translation);
        
        cv::RNG rng(-1);
        for (int i=0;i<num_translation;i++) {
            globalTranslations[i](0)=rng.uniform(-100.0,100.0);
            globalTranslations[i](1)=rng.uniform(-100.0,100.0);
            globalTranslations[i](2)=rng.uniform(-100.0,100.0);
            double angles[3]={rng.uniform(-10.0,10.0),rng.uniform(-10.0,10.0),rng.uniform(-10.0,10.0)};
            for (int r=0;r<3;r++) {
                angles[r]*=(CV_PI/180.0);
            }
            ceres::AngleAxisToRotationMatrix(angles,ceres::ColumnMajorAdapter3x3(globalRotations[i].data()));
        }
        
        std::vector<TranslationConstraint> constraints;
        
        for (int i=0;i<num_translation;i++) {
            TranslationConstraint constraint;
            
            constraint.variableIndex1=i;
            constraint.rotation1=globalRotations[i];
            
            for (int j=i+1;j<std::min(i+5,num_translation);j++) {
                constraint.variableIndex2=j;
                constraint.translation12=globalTranslations[j]-globalTranslations[i];
                constraint.weight=1.0;
                constraints.push_back(constraint);
            }
            
            if (i!=0) {
                continue;
            }
            
            constraint.variableIndex1=-1;
            constraint.variableIndex2=i;
            constraint.translation1=Eigen::Vector3d::Zero();
            constraint.translation12=globalTranslations[i];
            constraint.weight=1.0;
            constraints.push_back(constraint);
        }
        
        for (int i=0;i<num_translation;i++) {
            constraints[i].translation12(0)+=rng.uniform(-2.0,2.0);
            constraints[i].translation12(1)+=rng.uniform(-2.0,2.0);
            constraints[i].translation12(2)+=rng.uniform(-2.0,2.0);
        }
        
        std::vector<Eigen::Vector3d> results(num_translation,Eigen::Vector3d::Zero());
        this->solve(constraints,results);
        
        double error0=0;
        double error1=0;
        for (int i=0;i<num_translation;i++) {
            //std::cout<<i<<'\n'<<globalTranslations[i]-globalTranslations[0]<<'\n'<<results[i]-results[0]<<std::endl;
            error0+=(globalTranslations[i]-results[i]).norm();
            error1+=(globalTranslations[i]-globalTranslations[0]-results[i]+results[0]).norm();
        }
        
        std::cout<<error0<<' '<<error1<<std::endl;
    }
}
