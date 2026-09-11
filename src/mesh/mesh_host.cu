#include "mesh/mesh_type.h"

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <vector>

#include <fmt/format.h>

void Mesh::make_mesh_host(
    const std::vector<glm::vec3>& v, 
    const std::vector<int>& indices, 
    const std::vector<glm::vec3>& normals, 
    const std::vector<int>& normal_indices, 
    const std::vector<glm::vec2>& uvs, 
    const std::vector<int>& uv_indices
) {
    num_verts = v.size();
    num_triangles = indices.size() / 3;
    num_normals = normals.size();
    num_uvs = uvs.size();

    h_verts = v;
    h_triangles = std::vector<Triangle>();

    for(int i = 0; i < num_triangles; i++) {
        int nind[3];
        int uvind[3];
        int inds[3] = { indices[3*i], indices[3*i + 1], indices[3*i + 2] };
        if (normal_indices.size() > 0) {
            nind[0] = normal_indices[3*i];
            nind[1] = normal_indices[3*i + 1];
            nind[2] = normal_indices[3*i + 2];
        } else {
            nind[0] = -1;
            nind[1] = -1;
            nind[2] = -1;
        }

        if (uv_indices.size() > 0) {
            uvind[0] = uv_indices[3*i];
            uvind[1] = uv_indices[3*i + 1];
            uvind[2] = uv_indices[3*i + 2];
        } else {
            uvind[0] = -1;
            uvind[1] = -1;
            uvind[2] = -1;
        }

        h_triangles.push_back (
            Triangle (
                inds,
                nind,
                uvind
            )
        );
    }

    if (normals.size() > 0 && normal_indices.size() > 0) {
        h_normals = normals;
        has_normal_buffers = true;
    }

    if (uvs.size() > 0 && uv_indices.size() > 0) {
        h_uvs = uvs;
        has_uvs = true;
    }

    h_valid = true;
}
