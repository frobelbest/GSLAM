//
//  ImageGrids.hpp
//  STARCK
//
//  Created by Chaos on 4/7/16.
//  Copyright © 2016 Chaos. All rights reserved.
//

//
//  ImageGrids.h
//  SmallMotion
//
//  Created by chaos on 14-12-13.
//  Copyright (c) 2014年 chaos. All rights reserved.
//

#include "opencv2/core/core.hpp"
#pragma once

typedef struct TRIANGLE{
    int vertexIndex[3];
    int gridIndex;
}Triangle;

typedef struct EDGE{
    int vertexIndex[2];
}Edge;

class ImageGrids{
    
private:
    
    
    std::vector<cv::Point2d>  preparedGridVertices;
    std::vector<cv::Point2d>  warpedGridVertices;
    

    void initializeGridVertices(cv::Size gridSize,cv::Size sizeByVertex){
        
        originGridVertices.reserve(sizeByVertex.width*sizeByVertex.height);
        preparedGridVertices.resize(sizeByVertex.width*sizeByVertex.height);
        warpedGridVertices.resize(sizeByVertex.width*sizeByVertex.height);
        
        for (int i=0;i<sizeByVertex.height;i++) {
            for (int j=0;j<sizeByVertex.width;j++) {
                originGridVertices.push_back(cv::Point2f(j*gridSize.width,i*gridSize.height));
            }
        }
    }
    
public:
    std::vector<cv::Point2d>  originGridVertices;
    cv::Size sizeByGrid;
    cv::Size sizeByVertex;
    cv::Size gridSize;
    
    const std::vector<cv::Point2d>& getOriginGridVertices() const {return originGridVertices;};
    const std::vector<cv::Point2d>& getPreparedGridVertices() const {return preparedGridVertices;};
    
    std::vector<cv::Point2d>&       getWarpedGridVertices() {return warpedGridVertices;};
    
    void initialize(cv::Size _gridSize,cv::Size _sizeByGrid,cv::Size _sizeByVertex){
        
        initializeGridVertices(_gridSize,_sizeByVertex);
        
        sizeByGrid=_sizeByGrid;
        sizeByVertex=_sizeByVertex;
        gridSize=_gridSize;
        
    }
    
    
    void rotateAndNormalizePoint(const cv::Point2f &src,
                                 cv::Mat &dst,
                                 const std::vector<cv::Mat> &rotations,
                                 const cv::Mat &invK){
        
        double T[9];
        cv::Mat transform(3,3,CV_64FC1,T);
        
        int h0=src.y/gridSize.height;
        int h1=h0+1;
        
        int vIndex0=h0*sizeByVertex.width;
        int vIndex1=vIndex0+sizeByVertex.width;
        
        double dist0=src.y-originGridVertices[vIndex0].y;
        double dist1=originGridVertices[vIndex1].y-src.y;
        double dist =dist0+dist1;
        
        dist0=dist1/dist;
        dist1=1-dist0;
        
        transform=dist0*rotations[h0]+dist1*rotations[h1];
        transform=transform.t()*invK;
        
        dst=cv::Mat(3,1,CV_64FC1);
        double *data=(double*)dst.data;
        data[0]=  T[0]*src.x + T[1]*src.y + T[2];
        data[1]=  T[3]*src.x + T[4]*src.y + T[5];
        data[2]=  T[6]*src.x + T[7]*src.y + T[8];
        double normValue=cv::norm(dst,cv::NORM_L2);
        dst*=(1.0/normValue);
        
        return;
    }
    
    
    
    void prepareMeshByRow(const std::vector<cv::Mat> &transformArray){
        double T[9];
        cv::Mat transform(3,3,CV_64FC1,T);
        for (int h=0;h<sizeByVertex.height;h++) {
            int rowIndex=h*sizeByVertex.width;
            transformArray[h].convertTo(transform,CV_64FC1);
            for (int w=0;w<sizeByVertex.width;w++) {
                int index=rowIndex+w;
                double x=originGridVertices[index].x;
                double y=originGridVertices[index].y;
                double X = T[0]*x + T[1]*y + T[2];
                double Y = T[3]*x + T[4]*y + T[5];
                double W = T[6]*x + T[7]*y + T[8];
                W = W ? 1.0/W : 0;
                preparedGridVertices[index].x=X*W;
                preparedGridVertices[index].y=Y*W;
            }
        }
    }
    
    void prepareMeshByCopy(){
        std::copy(originGridVertices.begin(),originGridVertices.end(),preparedGridVertices.begin());
    };
    
    void release(){
        originGridVertices.clear();
        warpedGridVertices.clear();
    }
};
