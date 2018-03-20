//
//  IMU.hpp
//  STARCK
//
//  Created by Chaos on 4/6/16.
//  Copyright © 2016 Chaos. All rights reserved.
//
#include "opencv2/core/core.hpp"

typedef struct{
    double fl;
    double ox;
    double oy;
}Camera;


static void setCameraIntrisicMatrix(cv::Mat &K,cv::Mat &invK,const Camera &camera){
    double data[9]={0.0};
    data[0]=data[4]=camera.fl;
    data[8]=1.0;
    data[2]=camera.ox;
    data[5]=camera.oy;
    cv::Mat(3,3,CV_64FC1,data).copyTo(K);
    invert(K, invK);
}

static cv::Mat getSkewRotation(const cv::Scalar &angular){
    
    double theta = cv::norm(angular,cv::NORM_L2);
    cv::Scalar angularTheta;
    if (theta==0.0) {
        angularTheta=cv::Scalar(0);
    }else{
        angularTheta = angular/theta;
    }
    double SKEW_DATA[9] = {0.0, -angularTheta[2], angularTheta[0],
        angularTheta[2], 0.0, -angularTheta[1],
        -angularTheta[0], angularTheta[1], 0.0 };
    
    cv::Mat skew(3, 3, CV_64FC1, SKEW_DATA);
    cv::Mat result=cos(theta)*cv::Mat::eye(3,3,CV_64FC1)+sin(theta)*skew+(1 - cos(theta))*(skew*skew.t());
    return result;
}

typedef struct MOTION_DATA{
    double stamp;
    cv::Scalar anglev;
    cv::Scalar acc;
}MotionData;


class IMU{
    
private:
    
    std::vector<MotionData> motions;
    std::vector<double> gaps;
    
    
    void getDiffRotation(cv::Mat &rotation,
                         const int startGyroIndex,
                         const int endGyroIndex,
                         const double startStamp,
                         const double endStamp){
        
        double dt;
        if (startGyroIndex==endGyroIndex) {
            dt=endStamp-startStamp;
            rotation=rotation*getSkewRotation(motions[startGyroIndex].anglev * dt);
        }else{
            dt = motions[startGyroIndex+1].stamp-startStamp;
            rotation=rotation*getSkewRotation(motions[startGyroIndex].anglev * dt);
            
            for (int i=startGyroIndex+1;i<endGyroIndex;i++) {
                dt=gaps[i];
                rotation=rotation*getSkewRotation(motions[i].anglev * dt);
            }
            dt = endStamp - motions[endGyroIndex].stamp;
            rotation=rotation*getSkewRotation(motions[endGyroIndex].anglev * dt);
        }
    }
    
public:
    double ts;
    double wd[3];
    
    void loadImuData(const char* filename){
        
        motions.clear();
        gaps.clear();
        
        FILE *record = fopen(filename, "rb");
        
        bool withAcc;
        double tmpGyro[7];
        fscanf(record, "%lf %lf %lf %lf",
               &tmpGyro[0],
               &tmpGyro[1],
               &tmpGyro[2],
               &tmpGyro[3]);
        
        if (tmpGyro[3]<1){
            withAcc = true;
            fscanf(record, "%lf %lf %lf",
                   &tmpGyro[4],
                   &tmpGyro[5],
                   &tmpGyro[6]);
            
            
            cv::Scalar av(tmpGyro[3], tmpGyro[4], tmpGyro[5]);
            cv::Scalar ac(tmpGyro[0], tmpGyro[1], tmpGyro[2]);
            
            MotionData motion;
            motion.anglev=av;
            motion.acc=ac;
            motion.stamp=tmpGyro[6];
            
            motions.push_back(motion);
            
        }
        else{
            withAcc = false;
            /*tmpGyro[0] += param.wd[0];
            tmpGyro[1] += param.wd[1];
            tmpGyro[2] += param.wd[2];*/
            
            cv::Scalar av(tmpGyro[0], tmpGyro[1], tmpGyro[2]);
            
            MotionData motion;
            motion.anglev=av;
            motion.stamp=tmpGyro[3];
            motions.push_back(motion);
        }
        
        if (withAcc){
            double tmp[7];
            while (fscanf(record, "%lf %lf %lf %lf %lf %lf %lf",
                          &tmp[0],
                          &tmp[1],
                          &tmp[2],
                          &tmp[3],
                          &tmp[4],
                          &tmp[5],
                          &tmp[6]) != EOF){
                
                /*tmp[3] += param.wd[0];
                tmp[4] += param.wd[1];
                tmp[5] += param.wd[2];*/
                
                cv::Scalar av(tmp[3],tmp[4],tmp[5]);
                cv::Scalar ac(tmp[0],tmp[1],tmp[2]);
                
                MotionData motion;
                motion.anglev=av;
                motion.acc=ac;
                motion.stamp=tmp[6];
                
                motions.push_back(motion);
                
            }
        }else{
            double tmp[4];
            while (fscanf(record, "%lf %lf %lf %lf",
                          &tmp[0],
                          &tmp[1],
                          &tmp[2],
                          &tmp[3]) != EOF){
                
                /*tmp[0] += param.wd[0];
                tmp[1] += param.wd[1];
                tmp[2] += param.wd[2];*/
                
                cv::Scalar av(tmp[0], tmp[1], tmp[2]);
                
                MotionData motion;
                motion.anglev=av;
                motion.stamp=tmp[3];
                motions.push_back(motion);
            }
        }
        
        gaps.resize(motions.size());
        gaps[0]=0.0;
        for (int i=1;i<motions.size();i++) {
            gaps[i]=motions[i].stamp-motions[i-1].stamp;
            //printf("%f\n",gaps[i]);
        }
        fclose(record);
    }
    
    void updateMotionIndex(const double frameStamp,int &motionIndex){
        const int motionCount=motions.size();
        double motionStamp=motions[motionIndex].stamp;
        //printf("%d %f\n",motionCount,motionStamp);
        while (motionStamp<frameStamp&&motionIndex<motionCount-1) {
            motionIndex++;
            motionStamp=motions[motionIndex].stamp;
        }
        if (motionIndex>0) {
            motionIndex--;
        }
    }
    
    
    void synchronization(std::vector<double> &frameStamps,double ts){
        int frameCount = frameStamps.size();
        while (frameStamps[frameCount - 1] + ts>motions[motions.size()-1].stamp) {
            frameCount--;
        }
        frameStamps.resize(frameCount);
    }
    
    void getIntraFrameRotation(std::vector<cv::Mat> &rotations,
                               const std::vector<cv::Point2d> &originVertices,
                               cv::Size imageSize,
                               cv::Size sizeByVertex,
                               int &globalIndex,
                               double &globalStamp){
        
        //printf("%d %f\n",globalIndex,globalStamp);getchar();
        
        updateMotionIndex(globalStamp,globalIndex);
        rotations[0]=cv::Mat::eye(3,3,CV_64FC1);
        
        int preIndex=globalIndex;
        double preStamp=globalStamp;
        
        int rowIndex=globalIndex;
        double rowStamp=0;
        int rowVertexIndex=0;
        
        for (int i=1;i<sizeByVertex.height;i++) {
            rowVertexIndex+=sizeByVertex.width;
            double ratio=(double)originVertices[rowVertexIndex].y/((double)imageSize.height);
            rowStamp=globalStamp+ts*ratio;
            updateMotionIndex(rowStamp,rowIndex);
            rotations[i-1].copyTo(rotations[i]);
            getDiffRotation(rotations[i],preIndex,rowIndex,preStamp,rowStamp);
            preStamp=rowStamp;
            preIndex=rowIndex;
        }
        globalIndex=rowIndex;
        globalStamp=rowStamp;
    }
    
    void getInterFrameRotation(cv::Mat &rotation,int &nxtIndex,const double preStamp,const double nxtStamp){
        int preIndex=nxtIndex;
        updateMotionIndex(nxtStamp,nxtIndex);
        getDiffRotation(rotation,preIndex,nxtIndex,preStamp,nxtStamp);
    }
};
