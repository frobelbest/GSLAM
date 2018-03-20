//
//  Transform.cpp
//  GSLAM
//
//  Created by ctang on 9/5/16.
//  Copyright © 2016 ctang. All rights reserved.
//

#include "Transform.h"
#include "Geometry.h"
#include <iostream>

namespace GSLAM{
    
    Transform::Transform(){
        scale=1.0;
        rotation.setIdentity();
        translation.setZero();
    }
    
    void Transform::fromCameraToWorld(){
        
        
        
    }
    
    
    void Transform::fromWorldToCamera(){
        
        
    }
    
    void Transform::setEssentialMatrix(){
        setCrossMatrix(E,translation);
        E=rotation*E;
        E/=E(2,2);
    }
    
    Transform Transform::leftMultiply(const Transform& transform)const{
        Transform result;
        result.rotation=transform.rotation*rotation;
        result.translation=rotation.transpose()*transform.translation+translation;
        return  result;
    }
    
    Transform Transform::inverse()const{
        Transform result;
        result.rotation=rotation.transpose();
        result.translation=-rotation*translation;
        return result;
    }
    
    void Transform::display() const{
        std::cout<<rotation<<std::endl;
        std::cout<<translation<<std::endl;
    }
}
