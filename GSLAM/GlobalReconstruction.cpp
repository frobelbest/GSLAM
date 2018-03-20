
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
#include "engine.h"
#include "Drawer.h"

#include "theia/sfm/global_pose_estimation/robust_rotation_estimator.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/types.h"


#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>

//#include "common.h"

namespace GSLAM {
    
    class sim3PoseGraph{
        
    public:
        
        void optimizeSim3PoseGraph(const std::vector<SIM3Constraint> &constraints){
            
            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);
            g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverCholmod<g2o::BlockSolver_7_3::PoseMatrixType>();
            g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        
            solver->setUserLambdaInit(1e-16);
            optimizer.setAlgorithm(solver);
            
            vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(vpKFs.size());
            vector<g2o::VertexSim3Expmap*> vpVertices(vpKFs.size());

            for(size_t i=0, iend=vpKFs.size(); i<iend;i++){
                
                KeyFrame* pKF = vpKFs[i];
                g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();
                
                const int nIDi = pKF->mnId;
                assert(nIDi==pKF->mnId);
                
                
                Eigen::Matrix<double,3,3> Rcw   = pKF->pose.rotation;
                Eigen::Matrix<double,3,1> tcw   = -pKF->pose.rotation*pKF->pose.translation;
                double                    scale = pKF->scale;
                
                g2o::Sim3 Siw(Rcw,tcw,scale);
                
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
                
                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);
                
                if (i==0) {
                    VSim3->setFixed(true);
                }
                
                optimizer.addVertex(VSim3);
                vpVertices[nIDi]=VSim3;
            }
            
            const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

            for (int i=0;i<constraints.size();i++) {
                
                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                
                e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(constraints[i].variableIndex2)));
                e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(constraints[i].variableIndex1)));
                
                Eigen::Matrix<double,3,3> Rcw   =   constraints[i].rotation12;
                Eigen::Matrix<double,3,1> tcw   =  -constraints[i].rotation12*constraints[i].translation12;
                double                    scale =   1.0/constraints[i].scale12;
                
                
                //printf("%x %d %d\n",e,constraints[i].keyFrameIndex1,constraints[i].keyFrameIndex2);
                //cout<<Rcw<<endl;
                //cout<<tcw.transpose()<<endl;
                //cout<<constraints[i].scale12<<endl;
                
                g2o::Sim3 Siw(Rcw,tcw,scale);
                
                g2o::RobustKernelHuber* kernel=new g2o::RobustKernelHuber;
                kernel->setDelta(1.0);
                
                e->setRobustKernel(kernel);
                e->setMeasurement(Siw);
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
            
            optimizer.initializeOptimization();
            optimizer.optimize(20);
            
            for(size_t i=0;i<vpKFs.size();i++){
                
                KeyFrame* pKFi = vpKFs[i];
                const int nIDi = pKFi->mnId;
                g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
                g2o::Sim3 CorrectedSiw =  VSim3->estimate();
                Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = CorrectedSiw.translation();
                
                eigt=-eigR.transpose()*eigt;
                //std::cout<<eigt.transpose()<<endl;
                
                vpKFs[i]->pose.rotation=eigR;
                vpKFs[i]->pose.translation=eigt;
                vpKFs[i]->scale=CorrectedSiw.scale();
            }
        }
        std::vector<KeyFrame*> vpKFs;
    };
    
    class sim3PoseGraphEM{
    public:
        void optimizeSim3PoseGraph(std::vector<SIM3Constraint> &constraints){
            
            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);
            g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverCholmod<g2o::BlockSolver_7_3::PoseMatrixType>();
            g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
            
            solver->setUserLambdaInit(1e-16);
            optimizer.setAlgorithm(solver);
            
            vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(vpKFs.size());
            vector<g2o::VertexSim3Expmap*> vpVertices(vpKFs.size());
            
            
            for(size_t i=0, iend=vpKFs.size(); i<iend;i++){
                
                KeyFrame* pKF = vpKFs[i];
                g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();
                
                const int nIDi = pKF->mnId;
                assert(nIDi==pKF->mnId);
                
                
                Eigen::Matrix<double,3,3> Rcw   = pKF->pose.rotation;
                Eigen::Matrix<double,3,1> tcw   = -pKF->pose.rotation*pKF->pose.translation;
                double                    scale = pKF->scale;
                
                g2o::Sim3 Siw(Rcw,tcw,scale);
          
                //printf("%x\n",VSim3);
                //cout<<Rcw<<endl;
                
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
                
                VSim3->setId(nIDi);
                VSim3->setMarginalized(false);
                
                if (i==0) {
                    VSim3->setFixed(true);
                }
                
                optimizer.addVertex(VSim3);
                vpVertices[nIDi]=VSim3;
            }
            
            const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();
            std::vector<g2o::EdgeSim3*> edges(0);
            for (int i=0;i<constraints.size();i++) {
                
                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                
                e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(constraints[i].variableIndex1)));
                e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(constraints[i].variableIndex2)));
                
                Eigen::Matrix<double,3,3> Rcw   =   constraints[i].rotation12;
                Eigen::Matrix<double,3,1> tcw   =  -constraints[i].rotation12*constraints[i].translation12;
                double                    scale =   1.0/constraints[i].scale12;
                
                
                //printf("%x %d %d\n",e,constraints[i].keyFrameIndex1,constraints[i].keyFrameIndex2);
                //cout<<Rcw<<endl;
                //cout<<tcw.transpose()<<endl;
                //cout<<constraints[i].scale12<<endl;
                
                g2o::Sim3 Siw(Rcw,tcw,scale);
                
                //g2o::RobustKernelHuber* kernel=new g2o::RobustKernelHuber;
                //kernel->setDelta(1.0);
                //e->setRobustKernel(kernel);
                
                e->setMeasurement(Siw);
                e->information() = matLambda;
                optimizer.addEdge(e);
                edges.push_back(e);
                //e->computeError();
                /*printf("edge %x %x %d %d\n",optimizer.vertex(constraints[i].variableIndex1),
                       optimizer.vertex(constraints[i].variableIndex2),constraints[i].keyFrameIndex1,constraints[i].keyFrameIndex2);*/
            }
            
            double C=5.0;
            optimizer.initializeOptimization();
            optimizer.optimize(1);
            
            for ( int itr = 0; itr < 20; itr++ ) {
                // E step
                for(int i=0;i<edges.size();i++){
                    edges[i]->computeError();
                    double weight=(C*C) / (C*C+edges[i]->chi2()*edges[i]->chi2());
                    edges[i]->setInformation(matLambda*weight);
                }
                // M step
                optimizer.initializeOptimization();
                optimizer.optimize(1);
            }
            
            /*for(int i=0;i<edges.size();i++){
                edges[i]->computeError();
                constraints[i].weight=(C*C) / (C*C+edges[i]->chi2()*edges[i]->chi2());;
            }*/
            
            for(size_t i=0;i<vpKFs.size();i++){
                
                KeyFrame* pKFi = vpKFs[i];
                const int nIDi = pKFi->mnId;
                g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
                g2o::Sim3 CorrectedSiw =  VSim3->estimate();
                Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = CorrectedSiw.translation();
                eigt=-eigR.transpose()*eigt;
                
                vpKFs[i]->pose.rotation=eigR;
                vpKFs[i]->pose.translation=eigt;
                vpKFs[i]->scale=CorrectedSiw.scale();
            }
        }
        
        std::vector<KeyFrame*> vpKFs;
    };
    
    class GlobalRotationAveraging{
    public:
        
        Engine *ep;
        void initialize(){
            //Engine *ep = engOpen("");
            //assert(ep!=NULL);
        }
        
        void globalRotationAveraging(const int num_camera,
                                     std::vector<double> &cameraIds,
                                     std::vector<double> &relativeRotations,
                                     std::vector<double> &initialRotations){
            
            Engine *ep = engOpen(NULL);
            assert(ep!=NULL);
            
            mxArray *inArray[3];
            inArray[0]=mxCreateDoubleMatrix(cameraIds.size(),1,mxREAL);
            inArray[1]=mxCreateDoubleMatrix(relativeRotations.size(),1,mxREAL);
            inArray[2]=mxCreateDoubleMatrix(9*num_camera,1,mxREAL);
            
            
            memcpy(mxGetPr(inArray[0]),&cameraIds[0],cameraIds.size()*sizeof(double));
            memcpy(mxGetPr(inArray[1]),&relativeRotations[0],relativeRotations.size()*sizeof(double));
            
            engEvalString(ep,"clear all;");
            
            engPutVariable(ep, "cameraIds", inArray[0]);
            engPutVariable(ep, "relativeRotations", inArray[1]);
            engPutVariable(ep, "initialRotations", inArray[2]);
            
            engEvalString(ep, "cd '/Users/chaos/Desktop/Project/CuiLib'; globalR=GlobalRotationRegistration(relativeRotations,cameraIds);");
            mxArray *outArray=engGetVariable(ep,"globalR");
            double *globalRPtr=(double*)mxGetData(outArray);
            memcpy(&initialRotations[0],globalRPtr,sizeof(double)*initialRotations.size());
            
            mxDestroyArray(inArray[0]);
            mxDestroyArray(inArray[1]);
            mxDestroyArray(inArray[2]);
            mxDestroyArray(outArray);
            
            engClose(ep);
        }
        
        void estimateGlobalRotation(std::vector<RotationConstraint> &relativeTransforms,
                                    std::vector<Eigen::Matrix3d>   &globalRotations,
                                    int num_camera){
            
            int num_relative=relativeTransforms.size();
            std::vector<double> relativeRotations(9*num_relative);
            std::vector<double> cameraIds(2*num_relative);
            
            for(int i=0;i<num_relative;i++){
                
                int index1=relativeTransforms[i].variableIndex1;
                int index2=relativeTransforms[i].variableIndex2;
                
                cameraIds[2*i]=index1;
                cameraIds[2*i+1]=index2;
                
                Eigen::Matrix3d rotation=relativeTransforms[i].rotation12;
                rotation.transposeInPlace();
                memcpy(&relativeRotations[9*i],&rotation(0),9*sizeof(double));
            }
            
            std::vector<double> globalRotationsData(9*num_camera);
            globalRotationAveraging(num_camera,cameraIds,relativeRotations,globalRotationsData);
            
            globalRotations.resize(num_camera);
            for(int i=0;i<num_camera;i++){
                memcpy(&globalRotations[i](0),&globalRotationsData[9*i],9*sizeof(double));
                globalRotations[i].transposeInPlace();
            }
        }
        
        void release(){
            engClose(ep);
        }
    };

    
    void GlobalReconstruction::getScaleConstraint(KeyFrame* keyFrame1,
                                                  std::vector<ScaleConstraint>& scaleConstraints){
        
        for(map<KeyFrame*,std::vector<int> >::iterator mit=keyFrame1->mConnectedKeyFrameMatches.begin(),
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
            std::vector<double> scales;
            std::vector<int>& matches=mit->second;
            
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
                    
                    for(std::map<KeyFrame*,int>::iterator mit=keyFrame1->mvLocalMapPoints[i].gMP->measurements.begin(),
                        mend=keyFrame1->mvLocalMapPoints[i].gMP->measurements.end();
                        mit!=mend; mit++){
                        printf("%d %d\n",mit->first->outId,mit->second);
                    }
                    
                    for(std::map<KeyFrame*,int>::iterator mit=keyFrame2->mvLocalMapPoints[matches[i]].gMP->measurements.begin(),
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
                std::sort(scales.begin(),scales.end());
                constraint.value12=std::log(scales[scales.size()/2]);
            }else{
                continue;
            }
            
            constraint.weight=1.0;
            scaleConstraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::getRotationConstraint(KeyFrame* keyFrame1,
                                                     std::vector<RotationConstraint>& rotationConstraints){
        
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
            rotationConstraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::getTranslationConstraint(KeyFrame* keyFrame1,
                                                        std::vector<TranslationConstraint>& translationConstraints){
        
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
    
    void GlobalReconstruction::getSIM3Constraint(KeyFrame* keyFrame1,std::vector<SIM3Constraint>& sim3Constraints){
        
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
            
            std::vector<int>& matches=keyFrame1->mConnectedKeyFrameMatches[keyFrame2];
            std::vector<double> scales;
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
            
            std::sort(scales.begin(),scales.end());
            constraint.scale12=scales[scales.size()/2];
            
            sim3Constraints.push_back(constraint);
        }
    }
    
    void GlobalReconstruction::estimateScale(){
        
        std::vector<int> scaleIndex(keyFrames.size(),-1);
        std::vector<ScaleConstraint> scaleConstraints;
        
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
        
        std::vector<double> newScales;
        newScales.resize(nScales);
        for(int i=0;i<keyFrames.size();i++){
            if (scaleIndex[i]!=-1) {
                newScales[scaleIndex[i]]=keyFrames[i]->logScale;
            }
        }
        /*for (int i=0;i<scaleConstraints.size();i++) {
            
            std::cout<<scaleConstraints[i].keyFrameIndex1<<' '
                     <<scaleConstraints[i].keyFrameIndex2<<' '
                     <<scaleConstraints[i].variableIndex1<<' '
                     <<scaleConstraints[i].variableIndex2<<' '
                     <<scaleConstraints[i].value12<<std::endl;
        }
        for (int i=0;i<newScales.size();i++) {
            std::cout<<i<<' '<<newScales[i]<<std::endl;
        }*/
        globalScaleEstimation.maxIterations=10000;
        globalScaleEstimation.solve(scaleConstraints,newScales);
        
        for (int i=1;i<newScales.size();i++) {
            newScales[i]-=newScales[0];
        }
        newScales[0]=0.0;
        
        for (int i=0;i<keyFrames.size();i++) {
            keyFrames[i]->logScale=newScales[scaleIndex[i]];
            keyFrames[i]->scale=std::exp(keyFrames[i]->logScale);
            printf("%f\n",keyFrames[i]->scale);
        }
        
        static std::ofstream record("/Users/chaos/Desktop/debug/scales_error.txt");
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
                  <<diff<<std::endl;
        }
        //getchar();
    }
    
    void GlobalReconstruction::estimateRotation(std::vector<int> &rotationIndex){
        
        std::vector<RotationConstraint> rotationConstraints(0);
        std::vector<Eigen::Matrix3d> newRotations;
        
        /*RotationConstraint constraint;
         constraint.keyFrameIndex1=-1;
         constraint.keyFrameIndex2=0;
         constraint.variableIndex1=-1;
         constraint.variableIndex2=0;
         constraint.rotation1=Eigen::Matrix3d::Identity();
         constraint.rotation12=Eigen::Matrix3d::Identity();
         constraint.weight=1.0;
         rotationConstraints.push_back(constraint);*/
        
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
            //globalRotationEstimation.maxOuterIterations=1000;
            //globalRotationEstimation.maxInnerIterations=20;
            //globalRotationEstimation.solve(rotationConstraints,newRotations);
        }
        
        /*std::unordered_map<theia::ViewIdPair,theia::TwoViewInfo> view_pairs;
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
        }*/
        
        GlobalRotationAveraging rotationAveraging;
        rotationAveraging.initialize();
        rotationAveraging.estimateGlobalRotation(rotationConstraints,newRotations,
                                                 keyFrames.size());
        rotationAveraging.release();
        /*for (int i=1;i<newRotations.size();i++) {
            newRotations[i]=newRotations[i]*newRotations[0].transpose();
        }
        newRotations[0]=Eigen::Matrix3d::Identity();
        
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
        rotation_estimator.EstimateRotations(view_pairs,&orientations);
        for (int i=0;i<newRotations.size();i++) {
            ceres::AngleAxisToRotationMatrix(orientations[i].data(),newRotations[i].data());
        }*/
        
        
        for (int i=1;i<newRotations.size();i++) {
            newRotations[i]=newRotations[i]*newRotations[0].transpose();
        }
        newRotations[0]=Eigen::Matrix3d::Identity();
        for (int k=0;k<keyFrames.size();k++) {
            keyFrames[k]->pose.rotation=newRotations[rotationIndex[k]];
        }
    }
    
    void GlobalReconstruction::estimateRotationRobust(const std::vector<int> &rotationIndex){
        
        std::vector<RotationConstraint> rotationConstraints(0);
        std::vector<Eigen::Matrix3d> newRotations(keyFrames.size());

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
    
    void GlobalReconstruction::estimateTranslation(const std::vector<int> &translationIndex){
        
        std::vector<TranslationConstraint> translationConstraints;
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
            
            /*std::cout<<translationConstraints[i].variableIndex1<<' '
            <<translationConstraints[i].variableIndex2<<std::endl
            <<translationConstraints[i].translation12<<std::endl<<translationConstraints[i].rotation1<<std::endl;*/
        }
        
        std::vector<Eigen::Vector3d> newTranslations(keyFrames.size(),Eigen::Vector3d::Zero());
        globalTranslationEstimation.maxIterations=10000;
        globalTranslationEstimation.solve(translationConstraints,newTranslations);
        
        for (int i=1;i<newTranslations.size();i++) {
            newTranslations[i]-=newTranslations[0];
        }
        newTranslations[0]=Eigen::Vector3d::Zero();
        
        for (int k=0;k<keyFrames.size();k++) {
            keyFrames[k]->pose.translation=newTranslations[translationIndex[k]];
            std::cout<<keyFrames[k]->pose.translation.transpose()<<std::endl;
        }
        
        static std::ofstream record("/Users/chaos/Desktop/debug/trans_error.txt");
        
        
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
                  <<diff<<std::endl;

        }
    }
    
    void GlobalReconstruction::estimateSIM3(){
        
        std::vector<SIM3Constraint> rotationConstraints;
        int nRotations=0;
        std::vector<int> rotationIndex;
        rotationIndex.resize(keyFrames.size(),-1);
        
        sim3PoseGraph posegraph;
        //sim3PoseGraphEM posegraph;
        
        for (int k=0;k<keyFrames.size();k++) {
            getSIM3Constraint(keyFrames[k],rotationConstraints);
            
            for (int i=0;i<rotationConstraints.size();i++) {
                
                if (rotationIndex[rotationConstraints[i].keyFrameIndex1]==-1) {
                    rotationIndex[rotationConstraints[i].keyFrameIndex1]=nRotations;
                    nRotations++;
                }
                if (rotationIndex[rotationConstraints[i].keyFrameIndex2]==-1) {
                    rotationIndex[rotationConstraints[i].keyFrameIndex2]=nRotations;
                    nRotations++;
                }
                rotationConstraints[i].variableIndex1=rotationIndex[rotationConstraints[i].keyFrameIndex1];
                rotationConstraints[i].variableIndex2=rotationIndex[rotationConstraints[i].keyFrameIndex2];
                
                /*printf("%d %d %d %d\n",rotationConstraints[i].variableIndex1,rotationConstraints[i].keyFrameIndex1,
                                       rotationConstraints[i].variableIndex2,rotationConstraints[i].keyFrameIndex2);*/
                
            }
            
            //printf("%d %d\n",nRotations,keyFrames.size());
            //assert(nRotations<=keyFrames.size());
            //getchar();
            posegraph.vpKFs=this->keyFrames;
            posegraph.vpKFs.resize(nRotations);
            posegraph.optimizeSim3PoseGraph(rotationConstraints);
        }
        
        
        
        /*std::ofstream results("/Users/ctang/Desktop/debug/results.txt");
        for (int k=0;k<keyFrames.size();k++) {
            results<<keyFrames[k]->outId-1<<' '<<keyFrames[k]->pose.translation.transpose();
            for (int i1=0;i1<3;i1++) {
                for (int i2=0;i2<3;i2++) {
                    results<<' '<<keyFrames[k]->pose.rotation(i1,i2);
                }
            }
            results<<endl;
        }
        results.close();
        
        
        std::ofstream results0("/Users/ctang/Desktop/debug/connections.txt");
        for (int k=0;k<keyFrames.size();k++) {
            KeyFrame* keyFrame1=keyFrames[k];
            for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
                mend=keyFrame1->mConnectedKeyFramePoses.end();
                mit!=mend;
                mit++){
                KeyFrame* keyFrame2=mit->first;
                if (mit->second.scale==-1.0) {
                    results0<<keyFrame1->mnId<<' '<<keyFrame2->mnId<<' 2'<<endl;
                }else if (keyFrame2->nextKeyFramePtr==keyFrame1||keyFrame2->prevKeyFramePtr==keyFrame1) {
                    results0<<keyFrame1->mnId<<' '<<keyFrame2->mnId<<' 1'<<endl;
                }else{
                    results0<<keyFrame1->mnId<<' '<<keyFrame2->mnId<<' 0'<<endl;
                }
            }
        }
        results.close();
        
        
        std::ofstream results2("/Users/ctang/Desktop/debug/points.txt");
        for (int k=0;k<keyFrames.size();k++) {
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if(keyFrames[k]->mvLocalMapPoints[i].isEstimated){
                    
                    Eigen::Vector3d point3D=keyFrames[k]->mvLocalMapPoints[i].getPosition();
                    point3D*=keyFrames[k]->scale;
                    point3D=keyFrames[k]->pose.rotation.transpose()*point3D+keyFrames[k]->pose.translation;
                    
                    uchar* color=keyFrames[k]->mvLocalMapPoints[i].color;
                    results2<<point3D.transpose()<<' '<<(int)color[0]<<' '<<(int)color[1]<<' '<<(int)color[2]<<endl;
                }
            }
        }
        results2.close();*/

    }
    
    
    void GlobalReconstruction::addNewKeyFrame(KeyFrame* keyFrame){
        keyFrames.push_back(keyFrame);
    }
    
    void GlobalReconstruction::savePly(){
        
        std::ofstream results("/Users/chaos/Desktop/debug/key.txt");
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
        
        
        std::ofstream trajectories("/Users/chaos/Desktop/debug/tra.txt");
        for (int k=0;k<keyFrames.size();k++) {
            
            /*for (int i=0;i<keyFrames[k]->mvCloseFrames.size();i++) {
                results<<keyFrames[k]->mvCloseFrames[i].frameId;
                
                Eigen::Matrix3d rotation;
                rotation   =keyFrames[k]->mvCloseFrames[i].pose.rotation*keyFrames[k]->pose.rotation;
                
                Eigen::Vector3d translation;
                translation=keyFrames[k]->mvCloseFrames[i].pose.translation/keyFrames[k]->scale;
                translation=keyFrames[k]->pose.rotation.transpose()*translation;
                translation+=keyFrames[k]->pose.translation;
                
                trajectories<<keyFrames[k]->mvCloseFrames[i].frameId<<' '<<translation.transpose();
                for (int i1=0;i1<3;i1++) {
                    for (int i2=0;i2<3;i2++) {
                        trajectories<<' '<<rotation(i1,i2);
                    }
                }
                trajectories<<endl;
            }*/
            
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
        
        
        std::ofstream results0("/Users/chaos/Desktop/debug/connections.txt");
        for (int k=0;k<keyFrames.size();k++) {
            KeyFrame* keyFrame1=keyFrames[k];
            for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
                mend=keyFrame1->mConnectedKeyFramePoses.end();
                mit!=mend;
                mit++){
                KeyFrame* keyFrame2=mit->first;
                if (mit->second.scale==-1.0) {
                    results0<<keyFrame1->mnId<<' '<<keyFrame2->mnId<<" 2"<<endl;
                }else if (keyFrame2->nextKeyFramePtr==keyFrame1||keyFrame2->prevKeyFramePtr==keyFrame1) {
                    results0<<keyFrame1->mnId<<' '<<keyFrame2->mnId<<" 1"<<endl;
                }else{
                    results0<<keyFrame1->mnId<<' '<<keyFrame2->mnId<<" 0"<<endl;
                }
            }
        }
        results0.close();
        
        
        std::ofstream results1("/Users/chaos/Desktop/debug/connections2.txt");
        for (int k=0;k<keyFrames.size();k++) {
            KeyFrame* keyFrame1=keyFrames[k];
            for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
                mend=keyFrame1->mConnectedKeyFramePoses.end();
                mit!=mend;
                mit++){
                KeyFrame* keyFrame2=mit->first;
                if (mit->second.scale==-1.0) {
                    results1<<keyFrame1->frameId<<' '<<keyFrame2->frameId<<" 2"<<endl;
                }else if (keyFrame2->nextKeyFramePtr==keyFrame1||keyFrame2->prevKeyFramePtr==keyFrame1) {
                    results1<<keyFrame1->frameId<<' '<<keyFrame2->frameId<<" 1"<<endl;
                }else{
                    results1<<keyFrame1->frameId<<' '<<keyFrame2->frameId<<" 0"<<endl;
                }
            }
        }
        results1.close();
        
        
        
        
        
        //std::ofstream results2("/Users/ctang/Desktop/debug/points.txt");
        for (int k=0;k<keyFrames.size();k++) {
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                keyFrames[k]->mvLocalMapPoints[i].isEstimated=false;
            }
            
            for(map<KeyFrame*,std::vector<int> >::iterator mit=keyFrames[k]->mConnectedKeyFrameMatches.begin(),
                mend=keyFrames[k]->mConnectedKeyFrameMatches.end();
                mit!=mend;
                mit++){
                
                for (int i=0;i<mit->second.size();i++) {
                    if (mit->second[i]>=0) {
                        keyFrames[k]->mvLocalMapPoints[i].isEstimated=true;
                    }
                }
            }
            
            /*for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if(keyFrames[k]->mvLocalMapPoints[i].isEstimated){
                    
                    Eigen::Vector3d point3D=keyFrames[k]->mvLocalMapPoints[i].getPosition();
                    point3D/=keyFrames[k]->scale;
                    point3D=keyFrames[k]->pose.rotation.transpose()*point3D+keyFrames[k]->pose.translation;
                    
                    uchar* color=keyFrames[k]->mvLocalMapPoints[i].color;
                    results2<<point3D.transpose()<<' '<<(int)color[0]<<' '<<(int)color[1]<<' '<<(int)color[2]<<endl;
                }
            }*/
        }
        //results2.close();*/
        
        for (int nFrame=0;nFrame<keyFrames.size();nFrame++) {
            char name[200];
            sprintf(name,"/Users/chaos/Desktop/debug/points%d.txt",nFrame);
            std::ofstream results2(name);
            for(int k=0;k<=nFrame;k++){
                for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                    if(keyFrames[k]->mvLocalMapPoints[i].isEstimated){
                        
                        Eigen::Vector3d point3D=keyFrames[k]->mvLocalMapPoints[i].getPosition();
                        point3D/=keyFrames[k]->scale;
                        point3D=keyFrames[k]->pose.rotation.transpose()*point3D+keyFrames[k]->pose.translation;
                        
                        uchar* color=keyFrames[k]->mvLocalMapPoints[i].color;
                        results2<<point3D.transpose()<<' '<<(int)color[0]<<' '<<(int)color[1]<<' '<<(int)color[2]<<endl;
                    }
                }
            }
            results2.close();
        }
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
        
        cv::namedWindow("track");
        cv::moveWindow("track",720,0);
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
                
                char name[200];
                sprintf(name,"%s/image/frame%d.png",path,drawer.mFrameIndex+frameStart+1);
                printf(name);
                image=cv::imread(name);
                double ratio=720.0/image.cols;
                cv::resize(image,image,cv::Size(720,405));
               
                //printf("image %d %d\n",image.cols,image.rows);
                if ((drawer.mFrameIndex)==keyFrames[drawer.mKeyFrameIndex]->frameId) {
                    cv::RNG rng(-1);
                    for(int i=0;i<keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints.size();i++){
                        if(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount<=0){
                            continue;
                        }
                        //printf("%f %f %f\n",color.val[0],color.val[1],color[2]);
                        uchar *color=keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].color;
                        
                        int Size=std::min(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount,20);
                        Size=std::max(Size,3);
                        cv::circle(image,
                                   ratio*keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[0],ratio*Size,
                                   CV_RGB(245,211,40),2,CV_AA);
                        
                    }
                }else{
                    for(int i=0;i<keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints.size();i++){
                        
                        if(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount<=0){
                            continue;
                        }
                        
                        
                        if((drawer.mFrameIndex-keyFrames[drawer.mKeyFrameIndex]->frameId)
                           >=keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked.size()){
                            continue;
                        }
                        int localIndex=drawer.mFrameIndex-keyFrames[drawer.mKeyFrameIndex]->frameId;
                        
                        int Size=std::min(keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].measurementCount,20);
                        Size=std::max(Size,3);
                        uchar *color=keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].color;
                        cv::circle(image,
                                   ratio*keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[localIndex],
                                   ratio*Size,
                                   CV_RGB(245,211,40),2,CV_AA);
                        
                        for (int j=localIndex;j>std::max(localIndex-5,0);j--) {
                            cv::line(image,
                                     ratio*keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[j],
                                     ratio*keyFrames[drawer.mKeyFrameIndex]->mvLocalMapPoints[i].tracked[j-1],
                                     CV_RGB(245,211,40),2,CV_AA);
                        }
                    }
                }
                image.copyTo(preImage);
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
        
            char name[200];
            //sprintf(name,"%s/viewer/viewr%05d",path,drawer.mFrameIndex);
            //pangolin::SaveWindowOnRender(name);
            //sprintf(name,"%s/viewer/track%05d.png",path,drawer.mFrameIndex);
            //cv::imwrite(name,image);
            cv::imshow("tracking",image);
            image.release();
            pangolin::FinishFrame();
        }
    }
    
    bool myfunction (const LocalMapPoint* p1,const LocalMapPoint* p2) { return (p1->invdepth<p2->invdepth); }
    
    void GlobalReconstruction::topview(){
        
        /*for (int k=0;k<keyFrames.size();k++) {
            std::vector<LocalMapPoint*> localMapPoints;
            for (int i=0;i<keyFrames[k]->mvLocalMapPoints.size();i++) {
                if (keyFrames[k]->mvLocalMapPoints[i].isEstimated) {
                    localMapPoints.push_back(&keyFrames[k]->mvLocalMapPoints[i]);
                }
            }
            std::sort(localMapPoints.begin(),localMapPoints.end(),myfunction);
            
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
            
            char name[200];
            sprintf(name,"%s/viewer/top%05d",path,drawer.mFrameIndex);
            pangolin::SaveWindowOnRender(name);
            pangolin::FinishFrame();
            
        }
        
    }
}