#include "mesh.h"

void Mesh::make_mesh_host(std::vector<glm::vec3> v) {
    num_verts = v.size();
    h_verts = v;
    // cudaMalloc((void**)&d_verts, num_verts * sizeof(glm::vec3));
    // cudaMemcpy(d_verts, v.data(), num_verts * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    h_valid = true;
}

void Mesh::make_mesh_device() {
    cudaMalloc((void**)&d_verts, num_verts * sizeof(glm::vec3));
    cudaMemcpy(d_verts, h_verts.data(), num_verts * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    d_valid = true;
}

void Mesh::delete_mesh_device() {
    cudaFree(d_verts);
    d_valid = false;
}