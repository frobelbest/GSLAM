//
//  KeyFrameConnection.hpp
//  GSLAM
//
//  Created by ctang on 9/24/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#ifndef KeyFrameConnection_hpp
#define KeyFrameConnection_hpp

#include "KeyFrame.h"
#include "ORBmatcher.h"

namespace GSLAM {
    class KeyFrameConnection{
    public:
        
        ORBmatcher *matcherByBoW;
        ORBmatcher *matcherByTracking;
        ORBmatcher *matcherByProjection;
        
        ORBVocabulary* mpORBVocabulary;
        KeyFrameDatabase* keyFrameDatabase;
        
        void connectKeyFrame(KeyFrame* keyFrame1,KeyFrame* keyFrame2);
        void connectKeyFrame(KeyFrame* keyFrame1,KeyFrame* keyFrame2,std::vector<int>& matches12);
        int  connectKeyFrame(KeyFrame* keyFrame3,KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                             Transform& transform32,std::vector<int> &matches32,std::vector<int> &matches23);
        
        
        
        void connectLoop(KeyFrame* keyFrame2);
        int  connectLoop(KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                         Transform& transform12,std::vector<int> &matches12,std::vector<int> &matches21);
        
        
        int connectionThreshold;
    };
}
#endif /* KeyFrameConnection_hpp */
