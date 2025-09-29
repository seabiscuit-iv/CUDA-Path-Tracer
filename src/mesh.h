#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>

struct Mesh {
    bool h_valid = false;
    bool d_valid = false;

    std::vector<glm::vec3> h_verts;

    int num_verts = 0;
    glm::vec3* d_verts = nullptr;

    void make_mesh_host(std::vector<glm::vec3> v);
    void make_mesh_device();
    void delete_mesh_device();
};