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

    bvh.make_bvh(h_verts, h_indices);

    d_valid = true;
}

void Mesh::delete_mesh_device() {
    cudaFree(d_verts);
    cudaFree(d_indices);
    
    if (has_normal_buffers) {
        cudaFree(d_normals);
        cudaFree(d_normal_indices);
    }

    bvh.delete_bvh();

    d_valid = false;
}


void BVH::delete_bvh() {
    cudaFree(dev_bvh);
    initizalized = false;
}


// START AND END ARE INCLUSIVE
// we access indicies with 3 * start or 3 * end
BoundingBox fill_bvh(int idx, int start, int end, std::vector<BVHNode> &h_bvh, const std::vector<glm::vec3> &verts, const std::vector<int> &indices) {
    if (start == end) {
        // base case - single triangle
        h_bvh[idx] = BVHNode();
        h_bvh[idx].make_bvh_leaf_node(start);
        glm::vec3 local_tri[3] = {
            verts[indices[3 * start]],
            verts[indices[3 * start + 1]],
            verts[indices[3 * start + 2]]
        };
        return BoundingBox(local_tri, 3);
    }

    // partition [start_a ... end_a][start_b ... end_b]

    float log = glm::log2(float(end) - float(start));
    float floor = glm::floor(log);
    int mid = glm::pow(2.0f, floor);

    int start_a = start;
    int end_a = start + mid - 1;
    int start_b = end_a + 1;
    int end_b = end;

    BoundingBox bbox_a = fill_bvh(LEFT_NODE(idx), start_a, end_a, h_bvh, verts, indices);
    BoundingBox bbox_b = fill_bvh(RIGHT_NODE(idx), start_b, end_b, h_bvh, verts, indices);

    glm::vec3 combined_bbox[4] = {
        bbox_a.box_max,
        bbox_a.box_min,
        bbox_b.box_max,
        bbox_b.box_min,
    };

    h_bvh[idx] = BVHNode();
    auto fin_bbox = BoundingBox(combined_bbox, 4);
    h_bvh[idx].make_bvh_node(fin_bbox);

    return fin_bbox;
}



void BVH::make_bvh(std::vector<glm::vec3> verts, std::vector<int> indices) {
    // a binary tree with n leaf nodes must have 2n-1 nodes
    int num_leafs = indices.size() / 3; // 3 indices per triangle
    num_nodes = 2 * num_leafs - 1;

    std::vector<BVHNode> h_bvh(num_nodes);

    fill_bvh(0, 0, num_leafs-1, h_bvh, verts, indices);

    cudaMalloc((void**)&dev_bvh, num_nodes * sizeof(BVHNode));
    cudaMemcpy(dev_bvh, h_bvh.data(), num_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

    initizalized = true;
}

