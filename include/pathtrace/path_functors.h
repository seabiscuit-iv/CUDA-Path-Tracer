#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

// for stream compaction
struct path_terminated {
    __host__ __device__ bool operator()(PathSegment &path) const {
        return !path.kill;
    }
};

struct sort_materials {
    __host__ __device__ bool operator()(const ShadeableIntersection &sA, const ShadeableIntersection &sB) const {
        return sA.materialId < sB.materialId;
    }
};

struct sort_rays {
    const Geom* dev_mesh;

    sort_rays(const Geom* dev_geom)
        : dev_mesh(dev_geom)
    {}

    __device__ bool operator()(const PathSegment &path) const {
        Ray r = path.ray;

        r.origin = glm::vec3(dev_mesh->inverseTransform * glm::vec4(r.origin, 1.0f));
        r.direction = glm::vec3(dev_mesh->inverseTransform * glm::vec4(r.direction, 0.0f));

        #ifdef __CUDA_ARCH__
            r.inv_direction.x = __frcp_rn(r.direction.x);
            r.inv_direction.y = __frcp_rn(r.direction.y);
            r.inv_direction.z = __frcp_rn(r.direction.z);
        #else
            r.inv_direction = 1.0f / r.direction;
        #endif

        r.sign.x = (r.inv_direction.x < 0.0f) ? 1 : 0;
        r.sign.y = (r.inv_direction.y < 0.0f) ? 1 : 0;
        r.sign.z = (r.inv_direction.z < 0.0f) ? 1 : 0;

        float t;
        return dev_mesh->mesh.bvh.dev_bvh[0].box.RayBoxInterection(r, t);
    }
};
