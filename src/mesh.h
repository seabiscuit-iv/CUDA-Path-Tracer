#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>


class Ray;
class Triangle;

#define EPS 0.01f

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

        box_min = min - glm::vec3(EPS);
        box_max = max + glm::vec3(EPS);
    }

    BoundingBox() {
        box_min = glm::vec3(FLT_MAX);
        box_max = glm::vec3(-FLT_MAX);
    }

    void expandBox(BoundingBox bbox) {
        box_min.x = glm::min(box_min.x, bbox.box_min.x);
        box_min.y = glm::min(box_min.y, bbox.box_min.x);
        box_min.z = glm::min(box_min.z, bbox.box_min.x);

        box_max.x = glm::max(box_max.x, bbox.box_max.x);
        box_max.y = glm::max(box_max.y, bbox.box_max.x);
        box_max.z = glm::max(box_max.z, bbox.box_max.x);
    }

    void expandBoxByVec3(glm::vec3 p) {
        box_min.x = glm::min(box_min.x, p.x);
        box_min.y = glm::min(box_min.y, p.y);
        box_min.z = glm::min(box_min.z, p.z);

        box_max.x = glm::max(box_max.x, p.x);
        box_max.y = glm::max(box_max.y, p.y);
        box_max.z = glm::max(box_max.z, p.z);
    }

    float surfaceArea() const {
        float dx = box_max.x - box_min.x;
        float dy = box_max.y - box_min.y;
        float dz = box_max.z - box_min.z;
        return 2.0f * (dx*dy + dy*dz + dz*dx);
    }

    __host__ __device__
    bool RayBoxInterection(Ray ray);
};


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




struct Mesh {
    bool h_valid = false;
    bool d_valid = false;
    bool has_normal_buffers = false;

    std::vector<glm::vec3> h_verts;

    std::vector<glm::vec3> h_normals;
    
    std::vector<Triangle> h_triangles;

    int num_verts = 0;
    int num_triangles = 0;
    int num_normals = 0;

    glm::vec3* d_verts = nullptr;
    Triangle* d_triangles = nullptr;
    glm::vec3* d_normals = nullptr;

    BVH bvh;

    void make_mesh_host(std::vector<glm::vec3> v, std::vector<int> i, std::vector<glm::vec3> n, std::vector<int> ni);
    void make_mesh_device();
    void delete_mesh_device();
};
