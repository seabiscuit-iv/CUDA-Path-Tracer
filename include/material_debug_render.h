#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include <thrust/random.h>

__device__ void render_material_debug_mode (
    const Material& material, 
    PathSegment& path, 
    glm::vec3 materialColor, 
    glm::vec3 normal, 
    glm::vec3 normal_map,
    int idx,
    int num_paths,
    int iter,
    int depth,
    thrust::default_random_engine& rng,
    int material_debug_mode
);