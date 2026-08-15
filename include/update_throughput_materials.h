#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <thrust/random.h>

#include "config.h"
#include "sceneStructs.h"

__device__ void update_throughput_materials (
    const Material& material,
    PathSegment& path,
    int idx,
    int num_paths,
    int iter,
    int depth,
    thrust::default_random_engine& rng,
    glm::vec3 materialColor,
    glm::vec3 normal,
    float roughness,
    float metallic,
    bool is_specular
);