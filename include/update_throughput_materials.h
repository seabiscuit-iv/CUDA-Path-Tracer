#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "config.h"
#include "sceneStructs.h"

__device__ void update_throughput_materials (
    const Material& material,
    PathSegment& path,
    int idx,
    int num_paths,
    int iter,
    int depth,
    glm::vec3 materialColor,
    glm::vec3 normal,
    bool is_specular
);