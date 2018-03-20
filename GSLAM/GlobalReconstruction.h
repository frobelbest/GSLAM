//
//  GlobalReconstruction.h
//  GSLAM
//
//  Created by ctang on 9/8/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#ifndef GlobalReconstruction_h
#define GlobalReconstruction_h
#include <vector>
#include "Eigen/Dense"
#include "ceres/rotation.h"

namespace GSLAM{
    
    inline Eigen::Vector3d ComputeRelativeRotationError(const Eigen::Matrix3d& relative_rotation_matrix,
                                                        const Eigen::Matrix3d& rotation_matrix1,
                                                        const Eigen::Matrix3d& rotation_matrix2) {
        // Compute the relative rotation error.
        const Eigen::Matrix3d relative_rotation_matrix_error =rotation_matrix2.transpose()*relative_rotation_matrix*rotation_matrix1;
        Eigen::Vector3d relative_rotation_error;
        ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(relative_rotation_matrix_error.data()),
                                         relative_rotation_error.data());
        return relative_rotation_error;
    }
    
    inline void ApplyRotation(const Eigen::Vector3d& rotation_change,Eigen::Matrix3d& rotation_matrix) {
        // Convert to rotation matrices.
        Eigen::Matrix3d rotation_change_matrix;
        ceres::AngleAxisToRotationMatrix(rotation_change.data(),ceres::ColumnMajorAdapter3x3(rotation_change_matrix.data()));
        // Apply the rotation change.
        rotation_matrix = rotation_matrix*rotation_change_matrix;
    }
    
    inline int max3(int a, int b, int c){
        int m = a;
        (m < b) && (m = b); //these are not conditional statements.
        (m < c) && (m = c); //these are just boolean expressions.
        return m;
    }
    
    
    typedef struct{
        
        int variableIndex1;
        int variableIndex2;
        
        int keyFrameIndex1;
        int keyFrameIndex2;
        
        double value1;
        double value12;
        double weight;
        
        bool   isLoop;
        
    }ScaleConstraint;
    
    typedef struct{
        
        int variableIndex1;
        int variableIndex2;
        
        int keyFrameIndex1;
        int keyFrameIndex2;
        
        Eigen::Matrix3d rotation1;//if 1 is fixed
        Eigen::Matrix3d rotation12;
        
        double weight;
        bool   isLoop;
        
    }RotationConstraint;
    
    typedef struct{
        
        int variableIndex1;
        int variableIndex2;
        
        int keyFrameIndex1;
        int keyFrameIndex2;
        
        Eigen::Matrix3d rotation1;
        Eigen::Vector3d translation1;//if 1 is fixed
        Eigen::Vector3d translation12;
        
        double weight;
        bool   isLoop;
        
    }TranslationConstraint;
    
    typedef struct{
        
        int variableIndex1;
        int variableIndex2;
        
        int keyFrameIndex1;
        int keyFrameIndex2;
        
        Eigen::Matrix3d rotation12;
        Eigen::Vector3d translation12;
        double          scale12;
        
        double weight;
        bool   isLoop;
        
    }SIM3Constraint;
    
    
    class GlobalScaleEstimation{
        
    public:
        
        double	*objective;
        int    	*rowStart;
        int 	*column;
        double	*rowUpper;
        double	*rowLower;
        double	*columnUpper;
        double	*columnLower;
        double 	*elementByRow;
        
        int maxIterations;
        
        GlobalScaleEstimation(){
            objective=NULL;
            rowStart=NULL;
            column=NULL;
            rowUpper=NULL;
            rowLower=NULL;
            columnUpper=NULL;
            columnLower=NULL;
            elementByRow=NULL;
        }
        
        void solve(const std::vector<ScaleConstraint> &constraints,std::vector<double> &results);
        
        void test();
    };
    
    class GlobalRotationEstimation{
        
    public:
        
        double	*objective;
        int    	*rowStart;
        int 	*column;
        double	*rowUpper;
        double	*rowLower;
        double	*columnUpper;
        double	*columnLower;
        double 	*elementByRow;
        
        int     maxOuterIterations;
        int     maxInnerIterations;
        
        GlobalRotationEstimation(){
            objective=NULL;
            rowStart=NULL;
            column=NULL;
            rowUpper=NULL;
            rowLower=NULL;
            columnUpper=NULL;
            columnLower=NULL;
            elementByRow=NULL;
        }
        
        void solve(const std::vector<RotationConstraint> &constraints,std::vector<Eigen::Matrix3d>& results);
        
        void test();
        
    private:
        
    };
    
    class GlobalTranslationEstimation{
        
    public:
        
        double	*objective;
        int    	*rowStart;
        int 	*column;
        double	*rowUpper;
        double	*rowLower;
        double	*columnUpper;
        double	*columnLower;
        double 	*elementByRow;
        
        int     maxIterations;
        
        GlobalTranslationEstimation(){
            objective=NULL;
            rowStart=NULL;
            column=NULL;
            rowUpper=NULL;
            rowLower=NULL;
            columnUpper=NULL;
            columnLower=NULL;
            elementByRow=NULL;
        }
        
        void solve(const std::vector<TranslationConstraint> &constraints,std::vector<Eigen::Vector3d>& results);
        
        void solveComplex(const std::vector<TranslationConstraint> &constraints,std::vector<Eigen::Vector3d>& results);
        
        void test();
    };
    
    class KeyFrame;
    
    class GlobalReconstruction{
        
        std::vector<KeyFrame*> keyFrames;
        GlobalRotationEstimation    globalRotationEstimation;
        GlobalScaleEstimation       globalScaleEstimation;
        GlobalTranslationEstimation globalTranslationEstimation;
        
        void getScaleConstraint(KeyFrame* keyFrame1,std::vector<ScaleConstraint>& scaleConstraints);
        void getRotationConstraint(KeyFrame* keyFrame1,std::vector<RotationConstraint>& rotationConstraints);
        void getTranslationConstraint(KeyFrame* keyFrame1,std::vector<TranslationConstraint>& translationConstraints);
        void getSIM3Constraint(KeyFrame* keyFrame1,std::vector<SIM3Constraint>& constraints);
        
    public:
        char *path;
        int frameStart;
        GlobalReconstruction(){
            keyFrames.clear();
        }
        
        int scaleThreshold;
        void addNewKeyFrame(KeyFrame* keyFrame);
        void estimateScale();
        void estimateRotation(std::vector<int>& rotationIndex);
        void estimateRotationRobust(const std::vector<int>& rotationIndex);
        void estimateTranslation(const std::vector<int>& translationIndex);
        void estimateSIM3();
        void savePly();
        void visualize();
        void topview();
    };
}

#endif /* GlobalReconstruction_h */
