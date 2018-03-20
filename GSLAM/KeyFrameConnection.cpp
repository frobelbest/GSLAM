//
//  KeyFrameConnection.cpp
//  GSLAM
//
//  Created by ctang on 9/24/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "KeyFrameConnection.h"
#include "Estimation.h"

namespace GSLAM{
    
    void updateScore(ORBVocabulary* mpORBVocabulary,KeyFrame* keyFrame1,KeyFrame* keyFrame2){
        float score=mpORBVocabulary->score(keyFrame1->mBowVec,keyFrame2->mBowVec);
        if (score<keyFrame1->minScore) {
            keyFrame1->minScore=score;
        }
        if (score<keyFrame2->minScore) {
            keyFrame2->minScore=score;
        }
    }
    
    int KeyFrameConnection::connectLoop(KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                                        Transform& transform12,
                                        std::vector<int> &matches12,
                                        std::vector<int> &matches21){
    
        matches12.resize(keyFrame1->mvLocalMapPoints.size(),-1);
        matches21.resize(keyFrame2->mvLocalMapPoints.size(),-1);
        
        int nMatches12=matcherByBoW->SearchByBoW(keyFrame1,keyFrame2,matches12,matches21);
        printf("loop match0 %d %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12);
        
        if (nMatches12>connectionThreshold) {
            
            PnPEstimator pnp;
            pnp.prob=0.99;
            pnp.threshold=0.01;
            
            nMatches12=pnp.estimate(keyFrame1,keyFrame2,matches12,matches21,transform12);
            printf("loop pnp %d %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12);
            
            LocalBundleAdjustment localBA;
            localBA.projErrorThres=0.008;
            std::vector<double> errors;
            
            localBA.BAIterations=5;
            nMatches12=localBA.refineKeyFrameMatches(keyFrame1,keyFrame2,
                                                     transform12,matches12,matches21);
            printf("loop refine1 %d %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12);
            
            nMatches12=matcherByProjection->SearchByProjection(keyFrame1,keyFrame2,transform12,matches12,matches21,3,64);
            nMatches12=localBA.refineKeyFrameMatches(keyFrame1,keyFrame2,transform12,
                                                     matches12,matches21);
            
            printf("loop refine2 %d %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12);
            
            transform12.scale=-1.0;
        }
        
        return nMatches12;
    }
    
    void KeyFrameConnection::connectLoop(KeyFrame* keyFrame2){
        
        vector<KeyFrame*> vpCandidateKFs = keyFrameDatabase->DetectLoopCandidates(keyFrame2,keyFrame2->minScore*0.8);
        for (int i=0;i<vpCandidateKFs.size();i++) {
            KeyFrame* keyFrame1=vpCandidateKFs[i];
            
            //printf("loop detected %d %d %f %f\n",keyFrame1->frameId,keyFrame2->frameId,keyFrame1->minScore,keyFrame2->minScore);
            
            assert(!keyFrame1->mConnectedKeyFramePoses.count(keyFrame2)
                   &&!keyFrame2->mConnectedKeyFramePoses.count(keyFrame1));
            
            Transform transform12;
            std::vector<int> matches12,matches21;
            int nMatches12=connectLoop(keyFrame1,keyFrame2,transform12,matches12,matches21);
            
            Transform transform21;
            std::vector<int> _matches21,_matches12;
            int nMatches21=connectLoop(keyFrame2,keyFrame1,transform21,_matches21,_matches12);
            
            //printf("loop matched %d %d %d %d %f %f\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12,nMatches21,transform12.scale,transform21.scale);
            //printf("thres %d\n",connectionThreshold);
            if (nMatches12>connectionThreshold||nMatches21>connectionThreshold) {
                
                LocalBundleAdjustment localBA;
                localBA.BAIterations=5;
                localBA.projErrorThres=0.08;
                

                for (int i=0;i<matches12.size();i++) {
                    if (matches12[i]>=0&&_matches12[i]<0) {
                        _matches21[matches12[i]]=i;
                        //printf("%d %d added %d\n",keyFrame2->frameId,matches12[i],i);
                    }
                }
                
                for (int i=0;i<_matches21.size();i++) {
                    if (_matches21[i]>=0&&matches21[i]<0) {
                        matches12[_matches21[i]]=i;
                        //printf("%d %d added %d\n",keyFrame1->frameId,_matches21[i],i);
                    }
                }
                
                nMatches12=localBA.refineKeyFrameMatches(keyFrame1,keyFrame2,transform12,matches12,matches21);
                nMatches21=localBA.refineKeyFrameMatches(keyFrame2,keyFrame1,transform21,_matches21,_matches12);
                //cout<<transform12.translation.transpose();
                //cout<<transform21.translation.transpose();
                
                printf("loop refine3 %d %d %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12,nMatches21);
                
                if(nMatches12>connectionThreshold/2&&nMatches21>connectionThreshold/2){
                    keyFrame1->appendKeyFrame(keyFrame2,transform12,matches12);
                    localBA.refineKeyFrameConnection(keyFrame1);
                    updateScore(mpORBVocabulary,keyFrame1,keyFrame2);
                    //printf("loop connected %d %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches12);
                //}
                
                //if (nMatches21>connectionThreshold/2&&nMatches12>connectionThreshold/2) {
                    keyFrame2->appendKeyFrame(keyFrame1,transform21,_matches21);
                    localBA.refineKeyFrameConnection(keyFrame2);
                    updateScore(mpORBVocabulary,keyFrame1,keyFrame2);
                    //printf("loop connected %d %d %d\n",keyFrame2->frameId,keyFrame1->frameId,nMatches21);
                    
                    /*for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
                        mend=keyFrame1->mConnectedKeyFramePoses.end();
                        mit!=mend;
                        mit++){
                        
                        Transform transform32,transform23;
                        
                        std::vector<int> matches32,matches23;
                        std::vector<int> _matches32,_matches23;
                        
                        int nMatches32=connectKeyFrame(mit->first,keyFrame1,keyFrame2,transform32,matches32,matches23);
                        int nMatches23=connectKeyFrame(keyFrame2,keyFrame1,mit->first,transform23,_matches23,_matches32);
                        
                        if(nMatches32<connectionThreshold&&nMatches23<connectionThreshold){
                            continue;
                        }
                        
                        if (nMatches23==-1||nMatches32==-1) {
                            assert(0);
                        }
                        
                        for (int i=0;i<matches32.size();i++) {
                            if (matches32[i]>=0&&_matches32[i]<0) {
                                _matches23[matches32[i]]=i;
                            }
                        }
                        
                        for (int i=0;i<_matches23.size();i++) {
                            if (_matches23[i]>=0&&matches23[i]<0) {
                                matches32[_matches23[i]]=i;
                            }
                        }
                        
                        localBA.BAIterations=5;
                        nMatches32=localBA.refineKeyFrameMatches(mit->first,keyFrame2,transform32,matches32,matches23);
                        nMatches23=localBA.refineKeyFrameMatches(keyFrame2,mit->first,transform23,_matches23,_matches32);
                        
                        if(nMatches32>connectionThreshold/2&&nMatches23>connectionThreshold/2){
                            
                            mit->first->appendKeyFrame(keyFrame2,transform32,matches32);
                            localBA.refineKeyFrameConnection(mit->first);
                            
                            keyFrame2->appendKeyFrame(mit->first,transform23,_matches23);
                            localBA.refineKeyFrameConnection(keyFrame2);
                            
                            updateScore(mpORBVocabulary,keyFrame2,mit->first);
                        }
                    }
                    
                    
                    
                    for(map<KeyFrame*,Transform>::iterator mit=keyFrame2->mConnectedKeyFramePoses.begin(),
                        mend=keyFrame2->mConnectedKeyFramePoses.end();
                        mit!=mend;
                        mit++){
                        
                        KeyFrame* keyFrame3=mit->first;
                        
                        for(map<KeyFrame*,Transform>::iterator mit2=keyFrame3->mConnectedKeyFramePoses.begin(),
                            mend2=keyFrame3->mConnectedKeyFramePoses.end();
                            mit2!=mend2;
                            mit2++){
                            
                            if (keyFrame2->mConnectedKeyFramePoses.count(keyFrame3)) {
                                continue;
                            }
                            
                            Transform transform32,transform23;
                            std::vector<int> matches32,matches23;
                            std::vector<int> _matches32,_matches23;
                            
                            int nMatches32=connectKeyFrame(mit2->first,keyFrame3,keyFrame2,transform32,matches32,matches23);
                            int nMatches23=connectKeyFrame(keyFrame2,keyFrame3,mit2->first,transform23,_matches23,_matches32);
                            
                            if(nMatches32<connectionThreshold&&nMatches23<connectionThreshold){
                                continue;
                            }
                            
                            if (nMatches23==-1||nMatches32==-1) {
                                assert(0);
                            }
                            
                            for (int i=0;i<matches32.size();i++) {
                                if (matches32[i]>=0&&_matches32[i]<0) {
                                    _matches23[matches32[i]]=i;
                                }
                            }
                            
                            for (int i=0;i<_matches23.size();i++) {
                                if (_matches23[i]>=0&&matches23[i]<0) {
                                    matches32[_matches23[i]]=i;
                                }
                            }
                            
                            localBA.BAIterations=5;
                            nMatches32=localBA.refineKeyFrameMatches(mit->first,keyFrame2,transform32,matches32,matches23);
                            nMatches23=localBA.refineKeyFrameMatches(keyFrame2,mit->first,transform23,_matches23,_matches32);
                            

                            mit->first->appendKeyFrame(keyFrame2,transform32,matches32);
                            localBA.refineKeyFrameConnection(mit->first);
                            
                            keyFrame2->appendKeyFrame(mit->first,transform23,_matches23);
                            localBA.refineKeyFrameConnection(keyFrame2);
                            updateScore(mpORBVocabulary,keyFrame2,mit->first);
                        }
                    }*/
                }
            }
        }
    }
    

    
    int KeyFrameConnection::connectKeyFrame(KeyFrame* keyFrame3,KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                                            Transform& transform32,std::vector<int> &matches32,std::vector<int> &matches23){
        
        if (keyFrame3==keyFrame2||keyFrame3->mConnectedKeyFramePoses.count(keyFrame2)) {
            return -1;
        }
        
        const Transform transform31=keyFrame3->mConnectedKeyFramePoses[keyFrame1];
        const Transform transform13=keyFrame1->mConnectedKeyFramePoses[keyFrame3];
        
        matches32.resize(keyFrame3->mvLocalMapPoints.size(),-1);
        matches23.resize(keyFrame2->mvLocalMapPoints.size(),-1);
        
        if (!keyFrame3->mConnectedKeyFrameMatches.count(keyFrame1)
          ||!keyFrame1->mConnectedKeyFrameMatches.count(keyFrame2)) {
            assert(0);
            return -1;
        }
        
        const std::vector<int>& matches31=keyFrame3->mConnectedKeyFrameMatches[keyFrame1];
        const std::vector<int>& matches12=keyFrame1->mConnectedKeyFrameMatches[keyFrame2];
        
        int nMatches32=0;
        for (int i=0;i<keyFrame3->mvLocalMapPoints.size();i++) {
            int idx1=matches31[i];
            if (idx1<0) {
                continue;
            }
            //printf("%d %d %d %d %d %d\n",keyFrame3->frameId,keyFrame1->frameId,keyFrame2->frameId,i,idx1,matches12[idx1]);
            
            int index2=matches12[idx1];
            if (index2>=0) {
                assert(matches23[index2]==-1);
                matches32[i]=index2;
                matches23[index2]=i;
                nMatches32++;
            }
        }
        
        std::vector<double> scales;
        for (int i=0;i<matches31.size();i++) {
            if (matches31[i]<0) {
                continue;
            }else if(!keyFrame1->mvLocalMapPoints[matches31[i]].isEstimated){
                continue;
            }else if(!keyFrame3->mvLocalMapPoints[i].isEstimated){
                //printf("%d %d\n",keyFrame3->frameId,i);
                //std::cout<<keyFrame3->mvLocalMapPoints[i].measurementCount<<endl;;
                assert(0);
            }
            
            double distance3=(keyFrame3->mvLocalMapPoints[i].getPosition()-transform31.translation).norm();
            double distance1=keyFrame1->mvLocalMapPoints[matches31[i]].getPosition().norm();
            scales.push_back(distance3/distance1);
            
            distance3=keyFrame3->mvLocalMapPoints[i].getPosition().norm();
            distance1=(keyFrame1->mvLocalMapPoints[matches31[i]].getPosition()-transform13.translation).norm();
            scales.push_back(distance3/distance1);
        }
        
        double relativeScale;
        if (scales.size()>5) {
            std::sort(scales.begin(),scales.end());
            relativeScale=scales[scales.size()/2];
        }else{
            relativeScale=1.0;
        }
        
        
        
        Transform transform12=keyFrame1->mConnectedKeyFramePoses[keyFrame2];
        transform12.translation*=relativeScale;
        transform32=transform31.leftMultiply(transform12);
        
        LocalBundleAdjustment localBA;
        localBA.projErrorThres=0.008;
        localBA.BAIterations=5;
        printf("first pass");
        nMatches32=localBA.refineKeyFrameMatches(keyFrame3,keyFrame2,transform32,matches32,matches23);
        nMatches32=matcherByProjection->SearchByProjection(keyFrame3,keyFrame2,transform32,matches32,matches23,3,64);
        printf("second pass");
        nMatches32=localBA.refineKeyFrameMatches(keyFrame3,keyFrame2,transform32,matches32,matches23);
        
        return nMatches32;
    }
    
    void KeyFrameConnection::connectKeyFrame(KeyFrame* keyFrame1,KeyFrame* keyFrame2,
                                             std::vector<int> &matches21){
        std::vector<int> matches12;
        matches12.resize(keyFrame1->mvLocalMapPoints.size());
        matches21.resize(keyFrame2->mvLocalMapPoints.size());
        
        std::fill(matches12.begin(),matches12.end(),-1);
        std::fill(matches21.begin(),matches21.end(),-1);
        
        int nMatches1=matcherByTracking->SearchByTracking(keyFrame1,keyFrame2,matches12,matches21,80);
        int nMatches2=matcherByProjection->SearchByProjection(keyFrame1,keyFrame2,keyFrame1->mvLocalFrames.back().pose,
                                                              matches12,matches21,3,64);
        
        LocalBundleAdjustment localBA;
        localBA.projErrorThres=0.008;
        Transform pose12=keyFrame1->mvLocalFrames.back().pose;
        localBA.BAIterations=5;
        int nMatches=localBA.refineKeyFrameMatches(keyFrame1,keyFrame2,pose12,matches12,matches21);
    }
    
    void KeyFrameConnection::connectKeyFrame(KeyFrame* keyFrame1,KeyFrame* keyFrame2){
        
        std::vector<int> matches12,matches21;
        matches12.resize(keyFrame1->mvLocalMapPoints.size());
        matches21.resize(keyFrame2->mvLocalMapPoints.size());
        
        std::fill(matches12.begin(),matches12.end(),-1);
        std::fill(matches21.begin(),matches21.end(),-1);
        
        int nMatches1=matcherByTracking->SearchByTracking(keyFrame1,keyFrame2,matches12,matches21,80);
        int nMatches2=matcherByProjection->SearchByProjection(keyFrame1,keyFrame2,keyFrame1->mvLocalFrames.back().pose,
                                                              matches12,matches21,3,64);
        
        LocalBundleAdjustment localBA;
        localBA.projErrorThres=0.008;
        Transform pose12=keyFrame1->mvLocalFrames.back().pose;
        localBA.BAIterations=5;
        int nMatches=localBA.refineKeyFrameMatches(keyFrame1,keyFrame2,pose12,matches12,matches21);
        /*for (int i=0;i<matches12.size()-1;i++) {
            for (int j=i+1;j<matches12.size();j++) {
                if(matches12[i]>0&&matches12[j]>0){
                    assert(matches12[i]!=matches12[j]);
                }
            }
        }*/
        std::vector<int> _matches12,_matches21;
        _matches12.resize(keyFrame1->mvLocalMapPoints.size());
        _matches21.resize(keyFrame2->mvLocalMapPoints.size());
        
        std::fill(_matches12.begin(),_matches12.end(),-1);
        std::fill(_matches21.begin(),_matches21.end(),-1);
        
        std::vector<double> scales;
        for (int i=0;i<matches12.size();i++) {
            if (matches12[i]>=0&&keyFrame2->mvLocalMapPoints[matches12[i]].isEstimated) {
                
                _matches12[i]=matches12[i];
                _matches21[_matches12[i]]=i;
                
                double distance2=keyFrame2->mvLocalMapPoints[matches12[i]].getPosition().norm();
                double distance1=(keyFrame1->mvLocalMapPoints[i].getPosition()
                                  -keyFrame1->mvLocalFrames.back().pose.translation).norm();
                scales.push_back(distance2/distance1);
            }
        }
        std::sort(scales.begin(),scales.end());
        double scale=scales[scales.size()/2];
        Transform pose21=pose12.inverse();
        pose21.translation*=scale;
        
        localBA.BAIterations=5;
        int _nMatches2=localBA.refineKeyFrameMatches(keyFrame2,keyFrame1,pose21,_matches21,_matches12);
        _nMatches2=matcherByProjection->SearchByProjection(keyFrame2,keyFrame1,pose21,_matches21,_matches12,3,64);
        localBA.BAIterations=5;
        _nMatches2=localBA.refineKeyFrameMatches(keyFrame2,keyFrame1,pose21,_matches21,_matches12);
        
        
        /*printf("%d %d nMatches %d %d\n",keyFrame1->frameId,keyFrame2->frameId,nMatches,_nMatches2);
        for (int i=0;i<matches12.size()-1;i++) {
            for (int j=i+1;j<matches12.size();j++) {
                if(matches12[i]>0&&matches12[j]>0){
                    assert(matches12[i]!=matches12[j]);
                }
            }
        }
        
        for (int i=0;i<_matches21.size()-1;i++) {
            for (int j=i+1;j<_matches21.size();j++) {
                if(_matches21[i]>0&&_matches21[j]>0){
                    assert(_matches21[i]!=_matches21[j]);
                }
            }
        }*/
        
        for (int i=0;i<matches12.size();i++) {
            if (matches12[i]>=0&&_matches12[i]<0) {
                _matches21[matches12[i]]=i;
                //printf("%d %d added %d\n",keyFrame2->frameId,matches12[i],i);
            }
        }
        
        for (int i=0;i<_matches21.size();i++) {
            if (_matches21[i]>=0&&matches21[i]<0) {
                matches12[_matches21[i]]=i;
                //printf("%d %d added %d\n",keyFrame1->frameId,_matches21[i],i);
            }
        }
        
        /*for (int i=0;i<matches12.size()-1;i++) {
            for (int j=i+1;j<matches12.size();j++) {
                if(matches12[i]>0&&matches12[j]>0){
                    assert(matches12[i]!=matches12[j]);
                }
            }
        }
        
        for (int i=0;i<_matches21.size()-1;i++) {
            for (int j=i+1;j<_matches21.size();j++) {
                if(_matches21[i]>0&&_matches21[j]>0){
                    //printf("%d %d %d\n",keyFrame2->frameId,_matches21[i],matches21[j]);
                    assert(_matches21[i]!=_matches21[j]);
                }
            }
        }*/
        
        localBA.BAIterations=5;
        nMatches=localBA.refineKeyFrameMatches(keyFrame1,keyFrame2,pose12,matches12,matches21);
        _nMatches2=localBA.refineKeyFrameMatches(keyFrame2,keyFrame1,pose21,_matches21,_matches12);
        
        
        keyFrame1->appendKeyFrame(keyFrame2,keyFrame1->mvLocalFrames.back().pose,matches12);
        keyFrame1->nextKeyFramePtr=keyFrame2;
        
        keyFrame2->appendKeyFrame(keyFrame1,pose21,_matches21);
        keyFrame2->prevKeyFramePtr=keyFrame1;
        
        updateScore(mpORBVocabulary,keyFrame1,keyFrame2);
        
        localBA.projErrorThres=0.008;
        //second layer: connect keyframe2 to keyframe1's connection
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame1->mConnectedKeyFramePoses.begin(),
            mend=keyFrame1->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            
            Transform transform32,transform23;
            
            std::vector<int> matches32,matches23;
            std::vector<int> _matches32,_matches23;
            
            int nMatches32=connectKeyFrame(mit->first,keyFrame1,keyFrame2,transform32,matches32,matches23);
            int nMatches23=connectKeyFrame(keyFrame2,keyFrame1,mit->first,transform23,_matches23,_matches32);
            
            if(nMatches32<connectionThreshold&&nMatches23<connectionThreshold){
                continue;
            }
            
            if (nMatches23==-1||nMatches32==-1) {
                assert(0);
            }
            
            for (int i=0;i<matches32.size();i++) {
                if (matches32[i]>=0&&_matches32[i]<0) {
                    _matches23[matches32[i]]=i;
                }
            }
            
            for (int i=0;i<_matches23.size();i++) {
                if (_matches23[i]>=0&&matches23[i]<0) {
                    matches32[_matches23[i]]=i;
                }
            }
            
            localBA.BAIterations=5;
            nMatches32=localBA.refineKeyFrameMatches(mit->first,keyFrame2,transform32,matches32,matches23);
            nMatches23=localBA.refineKeyFrameMatches(keyFrame2,mit->first,transform23,_matches23,_matches32);
            
            if(nMatches32>connectionThreshold/2&&nMatches23>connectionThreshold/2){
                
                mit->first->appendKeyFrame(keyFrame2,transform32,matches32);
                localBA.refineKeyFrameConnection(mit->first);
                
                keyFrame2->appendKeyFrame(mit->first,transform23,_matches23);
                localBA.refineKeyFrameConnection(keyFrame2);
                
                updateScore(mpORBVocabulary,keyFrame2,mit->first);
            }
        }
        
        
        
        for(map<KeyFrame*,Transform>::iterator mit=keyFrame2->mConnectedKeyFramePoses.begin(),
            mend=keyFrame2->mConnectedKeyFramePoses.end();
            mit!=mend;
            mit++){
            
            KeyFrame* keyFrame3=mit->first;
            
            for(map<KeyFrame*,Transform>::iterator mit2=keyFrame3->mConnectedKeyFramePoses.begin(),
                mend2=keyFrame3->mConnectedKeyFramePoses.end();
                mit2!=mend2;
                mit2++){
                
                if (keyFrame2->mConnectedKeyFramePoses.count(keyFrame3)) {
                    continue;
                }
                
                Transform transform32,transform23;
                std::vector<int> matches32,matches23;
                std::vector<int> _matches32,_matches23;
                
                int nMatches32=connectKeyFrame(mit2->first,keyFrame3,keyFrame2,transform32,matches32,matches23);
                int nMatches23=connectKeyFrame(keyFrame2,keyFrame3,mit2->first,transform23,_matches23,_matches32);
                
                if(nMatches32<connectionThreshold&&nMatches23<connectionThreshold){
                    continue;
                }
                
                if (nMatches23==-1||nMatches32==-1) {
                    assert(0);
                }
                
                for (int i=0;i<matches32.size();i++) {
                    if (matches32[i]>=0&&_matches32[i]<0) {
                        _matches23[matches32[i]]=i;
                    }
                }
                
                for (int i=0;i<_matches23.size();i++) {
                    if (_matches23[i]>=0&&matches23[i]<0) {
                        matches32[_matches23[i]]=i;
                    }
                }
                
                localBA.BAIterations=5;
                nMatches32=localBA.refineKeyFrameMatches(mit->first,keyFrame2,transform32,matches32,matches23);
                nMatches23=localBA.refineKeyFrameMatches(keyFrame2,mit->first,transform23,_matches23,_matches32);
                
                
                mit->first->appendKeyFrame(keyFrame2,transform32,matches32);
                localBA.refineKeyFrameConnection(mit->first);
                
                keyFrame2->appendKeyFrame(mit->first,transform23,_matches23);
                localBA.refineKeyFrameConnection(keyFrame2);
                updateScore(mpORBVocabulary,keyFrame2,mit->first);
            }
        }
        
        connectionThreshold=80;
        connectLoop(keyFrame2);
    }
}