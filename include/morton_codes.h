#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "config.h"
#include "sceneStructs.h"

#define MORTON_INTERP_DIST 1.0f

__device__ inline uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ inline uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);

    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);

    return (xx << 2) | (yy << 1) | zz;
}

__device__ void normalizePoint(const glm::vec3& point, glm::vec3& out, const float scene_extent);

__device__ void normalizeDirection(const glm::vec3& dir, glm::vec3& out);

__device__ uint32_t rayMortonCode(const Ray& ray, const float scene_extent);

struct sort_rays_morton {
    const uint32_t* d_morton_codes;
    const bool* d_hit_geoms;

    sort_rays_morton(const uint32_t* d_m_c, const bool* d_h_g) :
        d_morton_codes(d_m_c),
        d_hit_geoms(d_h_g)
    {}

    __device__ bool operator()(int pA, int pB) const {
        bool b = !d_hit_geoms[pB];
        if (b || !d_hit_geoms[pA]) {
            return b;
        }

        return d_morton_codes[pA] < d_morton_codes[pB];
    }
};
