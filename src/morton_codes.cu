#include "morton_codes.h"

__device__ void normalizePoint(const glm::vec3& point, glm::vec3& out, const float scene_extent) {
    out.x = (point.x + scene_extent) / (2.0f * scene_extent);
    out.y = (point.y + scene_extent) / (2.0f * scene_extent);
    out.z = (point.z + scene_extent) / (2.0f * scene_extent);
}

__device__ void normalizeDirection(const glm::vec3& dir, glm::vec3& out) {
    out = (dir + glm::vec3(1.0f)) * 0.5f;
}

__device__ uint32_t rayMortonCode(const Ray& ray, const float scene_extent) {
    glm::vec3 p0_n, p1_n;
    
    normalizePoint(ray.origin, p0_n, scene_extent);
    
    glm::vec3 farPoint = ray.origin + (ray.direction * MORTON_INTERP_DIST); 
    normalizePoint(farPoint, p1_n, scene_extent);

    glm::vec3 midpoint = 0.5f * (p0_n + p1_n);

    return morton3D(midpoint.x, midpoint.y, midpoint.z);
}