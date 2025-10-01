#include "mesh.h"

void Mesh::make_mesh_host(std::vector<glm::vec3> v, std::vector<int> indices, std::vector<glm::vec3> normals, std::vector<int> normal_indices) {
    num_verts = v.size();
    num_indices = indices.size();
    num_normals = normals.size();
    num_normal_indices = normal_indices.size();

    h_verts = v;
    h_indices = indices;

    if (normals.size() > 0 && normal_indices.size() > 0) {
        h_normals = normals;
        h_normal_indices = normal_indices;
        has_normal_buffers = true;
    }

    h_valid = true;
}

void Mesh::make_mesh_device() {
    cudaMalloc((void**)&d_verts, num_verts * sizeof(glm::vec3));
    cudaMemcpy(d_verts, h_verts.data(), num_verts * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_indices, num_indices * sizeof(int));
    cudaMemcpy(d_indices, h_indices.data(), num_indices * sizeof(int), cudaMemcpyHostToDevice);
    
    if (has_normal_buffers) {
        cudaMalloc((void**)&d_normals, num_normals * sizeof(glm::vec3));
        cudaMemcpy(d_normals, h_normals.data(), num_normals * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        
        cudaMalloc((void**)&d_normal_indices, num_normal_indices * sizeof(int));
        cudaMemcpy(d_normal_indices, h_normal_indices.data(), num_normal_indices * sizeof(int), cudaMemcpyHostToDevice);
    }

    d_valid = true;
}

void Mesh::delete_mesh_device() {
    cudaFree(d_verts);
    cudaFree(d_indices);
    
    if (has_normal_buffers) {
        cudaFree(d_normals);
        cudaFree(d_normal_indices);
    }

    d_valid = false;
}