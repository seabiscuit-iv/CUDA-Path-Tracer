#include "mesh/mesh_type.h"

#include "sceneStructs.h"
#include "myoptix.h"

#include <glm/glm.hpp>
#include <vector>

#include <fmt/format.h>

void Mesh::make_mesh_device() {
    cudaMalloc((void**)&d_verts, num_verts * sizeof(glm::vec3));
    cudaMemcpy(d_verts, h_verts.data(), num_verts * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_triangles, num_triangles * sizeof(Triangle));
    cudaMemcpy(d_triangles, h_triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    
    if (has_normal_buffers) {
        cudaMalloc((void**)&d_normals, num_normals * sizeof(glm::vec3));
        cudaMemcpy(d_normals, h_normals.data(), num_normals * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }

    if (has_uvs) {
        cudaMalloc((void**)&d_uvs, num_uvs * sizeof(glm::vec2));
        cudaMemcpy(d_uvs, h_uvs.data(), num_uvs * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    }

    if (has_triangle_area_percentage_prefix) {
        float sum = h_triangle_area_percentage_prefix[h_triangle_area_percentage_prefix.size() - 1];

        cudaMalloc((void**)&d_triangle_area_percentage_prefix, num_triangles * sizeof(float));
        cudaMemcpy(d_triangle_area_percentage_prefix, h_triangle_area_percentage_prefix.data(), num_triangles * sizeof(float), cudaMemcpyHostToDevice);
    }
    else {
        fmt::println("ERROR: NO TRIANGLE AREA PERCENTAGE PREFIX GENERATED");
        exit(1);
    }

    bvh.make_bvh(h_verts, h_triangles);

    // optix
    build_optix_accel_structure(h_verts, d_verts, h_triangles, d_triangles, this->as_handle, this->d_as_output_buffer);

    d_valid = true;
}

void Mesh::make_mesh_device_copy(const Mesh& mesh) {
    // Guard for mesh aliasing
    if (num_verts != mesh.num_verts || num_triangles != mesh.num_triangles) {
        fmt::println("ERROR: Mesh {} cannot alias device buffers of {}: {}/{} verts, {}/{} triangles",
            label, mesh.label, num_verts, mesh.num_verts, num_triangles, mesh.num_triangles);
        exit(1);
    }

    num_normals = mesh.num_normals;
    num_uvs = mesh.num_uvs;

    has_normal_buffers = mesh.has_normal_buffers;
    has_uvs = mesh.has_uvs;
    has_triangle_area_percentage_prefix = mesh.has_triangle_area_percentage_prefix;

    d_verts = mesh.d_verts;
    d_triangles = mesh.d_triangles;
    d_normals = mesh.d_normals;
    d_uvs = mesh.d_uvs;
    d_triangle_area_percentage_prefix = mesh.d_triangle_area_percentage_prefix;
    bvh = mesh.bvh;
    as_handle = mesh.as_handle;
    d_as_output_buffer = mesh.d_as_output_buffer;
    d_valid = true;
    d_is_copy = true;
}


void Mesh::delete_mesh_device() {
    if (d_is_copy) {
        d_verts = nullptr;
        d_triangles = nullptr;
        d_normals = nullptr;
        d_uvs = nullptr;
        d_triangle_area_percentage_prefix = nullptr;
        d_valid = false;
        d_is_copy = false;
        return;
    }

    cudaFree(d_verts);
    cudaFree(d_triangles);
    
    if (has_normal_buffers) {
        cudaFree(d_normals);
    }

    if (has_uvs) {
        cudaFree(d_uvs);
    }

    if (has_triangle_area_percentage_prefix) {
        cudaFree(d_triangle_area_percentage_prefix);
    }

    cudaFree((void*)d_as_output_buffer);

    bvh.delete_bvh();

    d_valid = false;
}
