#include "mesh/bvh.h"
#include "mesh/bvh_build.h"

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <vector>
#include <numeric>

#include <fmt/format.h>

void BVH::delete_bvh() {
    cudaFree(dev_bvh);
    initizalized = false;
}


void BVH::make_bvh(std::vector<glm::vec3> verts, std::vector<Triangle> triangles) {
    // a binary tree with n leaf nodes must have 2n-1 nodes
    int num_leafs = triangles.size();
    num_nodes = 2 * num_leafs - 1;

    std::vector<BVHNode> h_bvh(num_nodes);
    
    std::vector<int> tri_indices(triangles.size());
    for(int i=0; i<triangles.size(); i++ ) {
        tri_indices[i] = i;
    }

    int allocated_length = 1;

    fill_bvh(0, 0, num_leafs-1, h_bvh, verts, triangles, tri_indices, allocated_length);

    cudaMalloc((void**)&dev_bvh, num_nodes * sizeof(BVHNode));
    cudaMemcpy(dev_bvh, h_bvh.data(), num_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

    initizalized = true;
}
