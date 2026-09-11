#pragma once

#include "mesh/bounding_box.h"

#include <glm/glm.hpp>
#include <vector>

struct Triangle;

struct BVHNode {
    bool isLeaf;
    BoundingBox box; // undefined if isLeaf == true
    int tri_index = -1; // indexes into d_indices and d_normal_indices

    int left_child = -1;

    void make_bvh_node(BoundingBox bbox, int lc) {
        isLeaf = false;
        box = bbox;
        tri_index = -1;
        left_child = lc;
    }

    void make_bvh_leaf_node(int idx) {
        isLeaf = true;
        this->tri_index = idx;
    }
};


struct BVH {
    bool initizalized = false;
    int num_nodes;
    BVHNode* dev_bvh;

    void make_bvh(std::vector<glm::vec3> v, std::vector<Triangle> i);
    void delete_bvh();
};
