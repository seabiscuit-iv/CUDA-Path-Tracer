#include "mesh.h"

void Mesh::make_mesh_host(std::vector<glm::vec3> v, std::vector<int> indices) {
    num_verts = v.size();
    num_indices = indices.size();

    h_verts = v;
    h_indices = indices;
    h_valid = true;
}

void Mesh::make_mesh_device() {
    cudaMalloc((void**)&d_verts, num_verts * sizeof(glm::vec3));
    cudaMemcpy(d_verts, h_verts.data(), num_verts * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_indices, num_indices * sizeof(int));
    cudaMemcpy(d_indices, h_indices.data(), num_indices * sizeof(int), cudaMemcpyHostToDevice);
    d_valid = true;
}

void Mesh::delete_mesh_device() {
    cudaFree(d_verts);
    cudaFree(d_indices);
    d_valid = false;
}