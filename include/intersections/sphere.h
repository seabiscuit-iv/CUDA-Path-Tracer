#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    const Geom &sphere,
    const Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
