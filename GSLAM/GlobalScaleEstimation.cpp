//
//  GlobalScaleEstimation.cpp
//  GSLAM
//
//  Created by ctang on 9/29/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "GlobalReconstruction.h"
#include "L1Solver.h"
#include "opencv2/core/core.hpp"

namespace GSLAM{
    
    void GlobalScaleEstimation::solve(const std::vector<ScaleConstraint>& constraints,std::vector<double>& results){
        
        //timeval time;
        //gettimeofday(&time, NULL);
        //long millis = (time.tv_sec * 1000) + (time.tv_usec / 1000);
        

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
        
        for (int i = 0; i <constraintCount; ++i){
            rowUpper[2*i]=constraints[i].value12;
            rowLower[2*i]=-COIN_DBL_MAX;
            rowUpper[2*i+1]=COIN_DBL_MAX;
            rowLower[2*i+1]=constraints[i].value12;
            
            if (constraints[i].variableIndex1==-1) {
                rowUpper[2*i]+=constraints[i].value1-results[constraints[i].variableIndex2];
                rowLower[2*i+1]+=constraints[i].value1-results[constraints[i].variableIndex2];
            }else{
                rowUpper[2*i]+=results[constraints[i].variableIndex1]-results[constraints[i].variableIndex2];
                rowLower[2*i+1]+=results[constraints[i].variableIndex1]-results[constraints[i].variableIndex2];
            }
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
        
        model.setMaximumIterations(this->maxIterations);
        model.loadProblem(byRow,columnLower,columnUpper,objective,rowLower,rowUpper);
        model.dual();
        const double *solutionPtr=model.primalColumnSolution();
        
        for (int i=0;i<variableCount;i++) {
            //printf("%d %f\n",i,solutionPtr[i]);
            results[i]+=solutionPtr[i];
        }
        
        //gettimeofday(&time, NULL);
        //long millis2 = (time.tv_sec * 1000) + (time.tv_usec / 1000);
        //getchar();
        //printf("%d ms\n",millis2-millis);
    }
    
    void GlobalScaleEstimation::test(){
        int num_scales=500;
        std::vector<double> globalScales(num_scales);
        cv::RNG rng(-1);
        for (int i=0;i<num_scales;i++) {
            globalScales[i]=rng.uniform(1.0,10.0);
        }
        
        std::vector<ScaleConstraint> constraints;
        for (int i=0;i<num_scales;i++) {
            ScaleConstraint constraint;
            constraint.variableIndex1=i;
            for (int j=i+1;j<std::min(i+5,num_scales);j++) {
                constraint.variableIndex2=j;
                constraint.value12=std::log(globalScales[j]/globalScales[i]);
                constraint.weight=1.0;
                constraints.push_back(constraint);
            }
            
            if (i<num_scales&&i>num_scales-100) {
                constraint.variableIndex1=-1;
                constraint.variableIndex2=i;
                constraint.value1=0.0;
                constraint.value12=std::log(globalScales[i]);
                constraint.weight=1.0;
                constraints.push_back(constraint);
            }
            
            if (i!=0) {
                continue;
            }
            constraint.variableIndex1=-1;
            constraint.variableIndex2=i;
            constraint.value1=0.0;
            constraint.value12=std::log(globalScales[i]);
            constraint.weight=1.0;
            constraints.push_back(constraint);
            

        }
        
        for (int i=0;i<constraints.size();i++) {
            constraints[i].value12+=std::log(rng.uniform(0.95,1.05));
        }
        
        int i=0;
        std::vector<ScaleConstraint> incrementalConstraints;
        std::vector<double> results2(6,0);
        
        for (int f=5;f<num_scales;f++) {
            for (;i<constraints.size();i++) {
                if (constraints[i].variableIndex2<=f) {
                    incrementalConstraints.push_back(constraints[i]);
                }else{
                    break;
                }
            }
            
            if (f==5) {
                this->solve(incrementalConstraints,results2);
            }else{
                std::vector<double> v;
                for (int j=incrementalConstraints.size()-1;j>0;j--) {
                    if (incrementalConstraints[j].variableIndex2==f) {
                        v.push_back(incrementalConstraints[j].value12+results2[incrementalConstraints[j].variableIndex1]);
                    }
                }
                std::sort(v.begin(),v.end());
                results2.push_back(v[v.size()/2]);
                this->maxIterations=1000;
                this->solve(incrementalConstraints,results2);
            }
        }
        
        this->maxIterations=10000;
        this->solve(incrementalConstraints,results2);
        
        this->maxIterations=10000;
        std::vector<double> results(num_scales,0);
        this->solve(constraints,results);
        results2.resize(results.size());
        for (int i=1;i<results.size();i++) {
            std::cout<<std::exp(results2[i]-results2[0])<<' '<<std::exp(results[i]-results[0])<<' '<<globalScales[i]/globalScales[0]<<std::endl;
        }
        
        double error0=0,error1=0,error2=0;
        for (int i=1;i<results.size();i++) {
            //error0+=std::abs(std::exp(results2[i]-results2[0])-globalScales[i]/globalScales[0]);
            //error1+=std::abs(std::exp(results[i]-results[0])-globalScales[i]/globalScales[0]);
            //error2+=std::abs(std::exp(results[i])-globalScales[i]);
            
            error0+=std::abs(results2[i]-results2[0]-std::log(globalScales[i]/globalScales[0]));
            error1+=std::abs(results[i]-results[0]-std::log(globalScales[i]/globalScales[0]));
            error2+=std::abs(results[i]-std::log(globalScales[i]));

        }
        std::cout<<error0/500<<' '<<error1/500<<' '<<error2/500<<std::endl;
    }
}