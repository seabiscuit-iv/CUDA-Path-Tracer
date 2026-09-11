#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ float boxIntersectionTest(
    const Geom &box,
    const Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
