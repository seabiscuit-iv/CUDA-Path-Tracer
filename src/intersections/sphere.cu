#include "intersections.h"
#include "stack.h"

__host__ __device__ float sphereIntersectionTest(
    const Geom &sphere,
    const Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    const float radius = 0.5f;

    // Transform ray into object space
    glm::vec3 ro = glm::vec3(multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f)));
    glm::vec3 rd = glm::vec3(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float b = glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - radius * radius;
    float disc = b * b - c;

    if (disc < 0.0f) {
        return -1.0f;
    }

    float sqrtDisc = sqrtf(disc);
    float t1 = -b - sqrtDisc;
    float t2 = -b + sqrtDisc;

    float t;
    if (t1 > 0.0f) {
        t = t1;
        outside = true;
    } else if (t2 > 0.0f) {
        t = t2;
        outside = false;
    } else {
        return -1.0f;
    }

    // Compute intersection in object space
    glm::vec3 p = ro + t * rd;

    // Transform back to world space
    intersectionPoint = glm::vec3(multiplyMV(sphere.transform, glm::vec4(p, 1.0f)));

    // Compute normal
    normal = glm::vec3(multiplyMV(sphere.invTranspose, glm::vec4(p, 0.0f)));
    normal = glm::normalize(normal);
    if (!outside) {
        normal = -normal;
    }

    // Return distance in world space (can also just return t if ray directions are normalized)
    return glm::length(r.origin - intersectionPoint);
}
