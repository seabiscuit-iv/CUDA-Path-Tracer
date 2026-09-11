#pragma once

#include "mesh/bvh.h"

#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <string>
#include <optix.h>

struct Triangle;

struct Mesh {
    std::string label;

    bool h_valid = false;
    bool d_valid = false;
    bool d_is_copy = false;
    bool has_normal_buffers = false;
    bool has_uvs = false;
    bool has_triangle_area_percentage_prefix = false;

    std::vector<glm::vec3> h_verts;

    std::vector<glm::vec3> h_normals;

    std::vector<glm::vec2> h_uvs;
    
    std::vector<Triangle> h_triangles;

    std::vector<float> h_triangle_area_percentage_prefix;

    int num_verts = 0;
    int num_triangles = 0;
    int num_normals = 0;
    int num_uvs = 0;

    glm::vec3* d_verts = nullptr;
    Triangle* d_triangles = nullptr;
    glm::vec3* d_normals = nullptr;
    glm::vec2* d_uvs = nullptr;
    float* d_triangle_area_percentage_prefix = nullptr;

    BVH bvh;

    OptixTraversableHandle as_handle;
    CUdeviceptr d_as_output_buffer;

    void make_mesh_host(
        const std::vector<glm::vec3>& v, 
        const std::vector<int>& indices, 
        const std::vector<glm::vec3>& normals, 
        const std::vector<int>& normal_indices,
        const std::vector<glm::vec2>& uvs, 
        const std::vector<int>& uv_indices
    );
    void make_mesh_device();
    void make_mesh_device_copy(const Mesh& mesh);
    void delete_mesh_device();
};
