//
//  main.cpp
//  STAR
//
//  Created by Chaos on 4/25/16.
//  Copyright © 2016 Chaos. All rights reserved.
//

#include "opencv2/imgproc/imgproc.hpp"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#include "System.h"
#include "GlobalReconstruction.h"
#include <sys/time.h>
#include <sys/stat.h>

int main(int argc, char **argv){
    
    printf("version no global ptr\n");
    system("exec rm -r /Users/chaos/Desktop/debug/*");
    
    argv[1]="/Users/chaos/Downloads/sequences/robot";
    argv[2]="/Users/chaos/Downloads/sequences/ORBvoc.txt";
    std::cout<<"process "<<argv[1]<<std::endl;
    
    char name[200];
    sprintf(name,"%s/viewer",argv[1]);
    
    mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //argv[2]="/Users/chaos/Downloads/ORB_SLAM2-master/Vocabulary/ORBvoc.txt";
    
    char loadname[200];
    sprintf(loadname,"%s/framestamp.txt",argv[1]);
    ifstream timestamps(loadname);
    
    //load imu
    sprintf(loadname,"%s/shake.mov",argv[1]);
    AVURLAsset *avAsset = [AVURLAsset URLAssetWithURL:
                           [NSURL fileURLWithPath:[NSString stringWithUTF8String:loadname]]
                                              options:nil];
    
    NSArray *videoTracks = [avAsset tracksWithMediaType:AVMediaTypeVideo];
    AVAssetTrack *videoTrack = [videoTracks objectAtIndex:0];
    
    NSError  *error;
    AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:avAsset error:&error];
    NSDictionary *options = [NSDictionary dictionaryWithObject:
                             [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
                                                        forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    
    AVAssetReaderTrackOutput *asset_reader_output = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack
                                                                                     outputSettings:options];
    [reader addOutput:asset_reader_output];
    [reader startReading];
    
    sprintf(loadname,"%s/config.yaml",argv[1]);
    GSLAM::System slamSystem(argv[2],loadname);
    slamSystem.path=argv[1];
    
    sprintf(loadname,"%s/gyro.txt",argv[1]);
    slamSystem.imu.loadImuData(loadname);

    
    int frameCount=0;
    std::ofstream record("/Users/chaos/Desktop/debug/time.txt");
    cv::Mat curImage,nextImage;
    while ( [reader status]==AVAssetReaderStatusReading ){
        
        CMSampleBufferRef sampleBuffer= [asset_reader_output copyNextSampleBuffer];
        
        if (sampleBuffer==NULL) {
            continue;
        }
        
        
        CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        CVPixelBufferLockBaseAddress(pixelBuffer,0);
        size_t width = CVPixelBufferGetWidth(pixelBuffer);
        size_t height = CVPixelBufferGetHeight(pixelBuffer);
        uint8_t *baseAddress =(uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer,0);
        
        
        uint8_t *baseAddress2 =(uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer,1);
        cv::Mat uv(height/2,width/2,CV_8UC2,baseAddress2,0);

        //create image with timestamps;
        cv::Mat image(height,width,CV_8UC1,baseAddress,0);
        
        std::vector<cv::Mat> yuv(3);
        cv::split(uv,&yuv[1]);
        cv::resize(image,yuv[0],cv::Size(width/2,height/2));
        cv::Mat rgb;
        cv::merge((const cv::Mat*)(&yuv[0]),3,rgb);
        cv::cvtColor(rgb,rgb,CV_YCrCb2RGB);
        
        double timestamp;
        timestamps>>timestamp;
        
        if(frameCount==slamSystem.frameStart){
            
            slamSystem.colorImage=&rgb;
            image.copyTo(curImage);
            
        }else if(frameCount>slamSystem.frameStart){
            image.copyTo(nextImage);
            slamSystem.preloadImage=&nextImage;
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            GSLAM::Transform pose=slamSystem.Track(curImage,timestamp,frameCount);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            record<<ttrack<<std::endl;
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer,0);
        CFRelease(sampleBuffer);
        sampleBuffer = NULL;
        frameCount++;
        
        if(frameCount>slamSystem.frameEnd){
            break;
        }
    }

    record.close();
    slamSystem.finish();
    cout<<"processed "<<frameCount<<" frames"<<endl;
    [reader cancelReading];
    timestamps.close();
    return 0;
}
