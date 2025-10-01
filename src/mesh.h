#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>

struct Mesh {
    bool h_valid = false;
    bool d_valid = false;
    bool has_normal_buffers = false;

    std::vector<glm::vec3> h_verts;
    std::vector<int> h_indices;

    std::vector<glm::vec3> h_normals;
    std::vector<int> h_normal_indices;

    int num_verts = 0;
    int num_indices = 0;

    int num_normals = 0;
    int num_normal_indices = 0;

    glm::vec3* d_verts = nullptr;
    int* d_indices = nullptr;
    glm::vec3* d_normals = nullptr;
    int* d_normal_indices = nullptr;

    void make_mesh_host(std::vector<glm::vec3> v, std::vector<int> i, std::vector<glm::vec3> n, std::vector<int> ni);
    void make_mesh_device();
    void delete_mesh_device();
};