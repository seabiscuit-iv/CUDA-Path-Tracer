#pragma once

#include <optix.h>
#include <vector>
#include <glm/glm.hpp>

#include "sceneStructs.h"

void build_optix_accel_structure(
    const std::vector<glm::vec3>& h_verts, 
    const glm::vec3* d_verts, 
    const std::vector<Triangle>& h_triangles, 
    const Triangle* d_triangles,
    OptixTraversableHandle& out_handle,
    CUdeviceptr& out_buffer
); 
