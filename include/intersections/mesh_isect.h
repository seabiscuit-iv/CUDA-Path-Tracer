#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

__device__ float meshIntersectionTest(    
    const Geom &mesh,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside );
