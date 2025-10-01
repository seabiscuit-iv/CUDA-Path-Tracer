#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>


struct BoundingBox {
    glm::vec3 box_min; // mins of x, y, z
    glm::vec3 box_max; // maxs of x, y, z

    BoundingBox(glm::vec3 verts[], int length) {
        glm::vec3 min = verts[0];
        glm::vec3 max = verts[0];

        for (int i = 1; i < length; i++) {
            min.x = glm::min(min.x, verts[i].x);
            min.y = glm::min(min.y, verts[i].y);
            min.z = glm::min(min.z, verts[i].z);

            max.x = glm::max(max.x, verts[i].x);
            max.y = glm::max(max.y, verts[i].y);
            max.z = glm::max(max.z, verts[i].z);
        }

        box_min = min;
        box_max = max;
    }

    BoundingBox() {
        box_min = glm::vec3(0.0);
        box_max = glm::vec3(0.0);
    }
};




struct BVHNode {
    bool isLeaf;
    BoundingBox box; // undefined if isLeaf == true
    int tri_indices = -1; // indexes into d_indices and d_normal_indices


    void make_bvh_node(BoundingBox bbox) {
        isLeaf = false;
        box = bbox;
        tri_indices = -1;
    }

    void make_bvh_leaf_node(int idx) {
        isLeaf = true;
        this->tri_indices = idx;
    }
};



#define LEFT_NODE(x) 2*x + 1
#define RIGHT_NODE(x) 2*x + 2

struct BVH {
    bool initizalized = false;
    int num_nodes;
    BVHNode* dev_bvh;

    void make_bvh(std::vector<glm::vec3> v, std::vector<int> i);
    void delete_bvh();
};




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

    BVH bvh;

    void make_mesh_host(std::vector<glm::vec3> v, std::vector<int> i, std::vector<glm::vec3> n, std::vector<int> ni);
    void make_mesh_device();
    void delete_mesh_device();
};
