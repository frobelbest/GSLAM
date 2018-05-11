//
//  System.h
//  GSLAM
//
//  Created by ctang on 8/30/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "opencv2/core/core.hpp"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "KeyFrameConnection.h"

#include "ORBmatcher.h"
#include "KLT.h"

//#include "IMU.hpp"
#include "ImageGrids.hpp"
#include "GlobalReconstruction.h"
#include <tuple>
#include <string>

using namespace std;




namespace GSLAM{
    
    class PyramidBuffer{
    public:
        std::thread preloadThread;
        std::vector<cv::Mat>* ptrs[3];
        std::vector<cv::Mat>  buffers[3];
        
        void initialize(){
            
            buffers[0]=std::vector<cv::Mat>();
            buffers[1]=std::vector<cv::Mat>();
            buffers[2]=std::vector<cv::Mat>();
            
            ptrs[0]=&buffers[0];
            ptrs[1]=&buffers[1];
            ptrs[2]=&buffers[2];
        }
        
        void next(){
            if(preloadThread.joinable()){
                preloadThread.join();
            }
            std::vector<cv::Mat>* tmp=ptrs[0];
            ptrs[0]=ptrs[1];
            ptrs[1]=ptrs[2];
            ptrs[2]=tmp;
        }
    };
    
    
    class System{
        
    public:
        
        System(const string &strVocFile, const string &strSettingsFile);
        
        Transform Track(cv::Mat &im, const double &timestamp,const int outIndex);
        
        void finish();
        
        IMU imu;
        
        cv::Mat *colorImage;
        char* path;
        int frameStart;
        int frameEnd;
        
        
        KeyFrame* frontKeyFrame;
        KeyFrame* preKeyFrame;
        SLAMSettings slamSettings;
        KeyFrameConnection keyFrameConnector;
        ORBVocabulary* mpVocabulary;
        KeyFrameDatabase* mpKeyFrameDatabase;
        GlobalReconstruction globalReconstruction;
        
        cv::Mat* preloadImage;
        
        
        
    private:
        

        
        //
        int frameId;
        
        //for orb

        
        cv::Mat mK;
        cv::Mat mKInv;
        cv::Mat mDistCoef;
        double  mScaleFactor;
        
        ORBextractor *mpORBExtractor;
        ORBmatcher   *orbMatcher;
        
        //for klt tracking
        PyramidBuffer pyramidBuffers;
        std::vector<cv::Mat> *pyramid1Ptr,*pyramid2Ptr,*pyramid3Ptr,*pyramidTmpPtr,pyramid1,pyramid2,pyramid3;
        KLT_TrackingContext  trackingContext;
        KLT_FeatureList featureList;
        
        //for geometry
        cv::Mat cvInvK;
        Eigen::Matrix3d K,invK;
        std::list<KeyFrame*>        activeKeyFrames;
        //KeyFrame* frontKeyFrame;
        //KeyFrame* preKeyFrame;
        std::vector<int> matchesToPre;
        
        //parameters
        CameraSettings cameraSettings;
        ORBExtractorSettings orbExtractorSettings;
        ///SLAMSettings slamSettings;
        
        //temporal parameter for imu
        ImageGrids grids;
        std::vector<cv::Mat> rotations;
        cv::Mat rotation;
        int frameIndex;
        double frameStamp;
        double preFrameStamp;
        cv::Size frameSize;
        std::vector<cv::Mat>   globalRotations;
        cv::Size sizeByVertex;
        
        //keyframe connection
        //KeyFrameConnection keyFrameConnector;
        
        //globalReconstruction
        
        
        
        //motion lighting compensate
        ORBextractor *mpORBExtractorFrame;
        Frame *lightFramePre,*lightFrameCur;
    };
}
