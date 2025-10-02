#include "mesh.h"
#include <stack>
#include "stack.h"
#include "sceneStructs.h"
#include <algorithm>
#include <glm/glm.hpp>

void Mesh::make_mesh_host(std::vector<glm::vec3> v, std::vector<int> indices, std::vector<glm::vec3> normals, std::vector<int> normal_indices) {
    num_verts = v.size();
    num_triangles = indices.size() / 3;
    num_normals = normals.size();

    h_verts = v;
    h_triangles = std::vector<Triangle>();

    for(int i = 0; i < num_triangles; i++) {
        int nind[3];
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

        h_triangles.push_back (
            Triangle (
                inds,
                nind
            )
        );
    }

    if (normals.size() > 0 && normal_indices.size() > 0) {
        h_normals = normals;
        has_normal_buffers = true;
    }

    h_valid = true;
}

void Mesh::make_mesh_device() {
    cudaMalloc((void**)&d_verts, num_verts * sizeof(glm::vec3));
    cudaMemcpy(d_verts, h_verts.data(), num_verts * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_triangles, num_triangles * sizeof(Triangle));
    cudaMemcpy(d_triangles, h_triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    
    if (has_normal_buffers) {
        cudaMalloc((void**)&d_normals, num_normals * sizeof(glm::vec3));
        cudaMemcpy(d_normals, h_normals.data(), num_normals * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }

    bvh.make_bvh(h_verts, h_triangles);

    d_valid = true;
}

void Mesh::delete_mesh_device() {
    cudaFree(d_verts);
    cudaFree(d_triangles);
    
    if (has_normal_buffers) {
        cudaFree(d_normals);
    }

    bvh.delete_bvh();

    d_valid = false;
}


void BVH::delete_bvh() {
    cudaFree(dev_bvh);
    initizalized = false;
}


// START AND END ARE INCLUSIVE
BoundingBox fill_bvh(int idx, int start, int end, std::vector<BVHNode> &h_bvh, const std::vector<glm::vec3> &verts, const std::vector<Triangle> &triangles, std::vector<int> &tri_indices, int &allocated_length) {
    if (start == end) {
        // base case - single triangle
        h_bvh[idx] = BVHNode();
        h_bvh[idx].make_bvh_leaf_node(tri_indices[start]);
        glm::vec3 local_tri[3] = {
            verts[triangles[tri_indices[start]].v_indices[0]],
            verts[triangles[tri_indices[start]].v_indices[1]],
            verts[triangles[tri_indices[start]].v_indices[2]]
        };
        return BoundingBox(local_tri, 3);
    }

    // partition [start_a ... end_a][start_b ... end_b]

    std::vector<glm::vec3> centroids;

    for(auto &triangle : triangles) {
        glm::vec3 centroid(0.0f);
        centroid += verts[triangle.v_indices[0]];
        centroid += verts[triangle.v_indices[1]];
        centroid += verts[triangle.v_indices[2]];
        centroid /= 3.0f;
        centroids.push_back(centroid);
    }

    float x_min = FLT_MAX, x_max = -FLT_MAX, y_min = FLT_MAX, y_max = -FLT_MAX, z_min = FLT_MAX, z_max = -FLT_MAX;

    for (auto &centroid : centroids) {
        x_min = glm::min(x_min, centroid.x);
        x_max = glm::max(x_max, centroid.x);
        
        y_min = glm::min(y_min, centroid.y);
        y_max = glm::max(y_max, centroid.y);

        z_min = glm::min(z_min, centroid.z);
        z_max = glm::max(z_max, centroid.z);
    }

    float x_dist = x_max - x_min;
    float y_dist = y_max - y_min;
    float z_dist = z_max - z_min;

    
    std::sort(tri_indices.begin() + start, tri_indices.begin() + end + 1, 
        [&](int a, int b) {
            glm::vec3 cA = (verts[triangles[a].v_indices[0]] + verts[triangles[a].v_indices[1]] + verts[triangles[a].v_indices[2]]) / 3.0f;
            glm::vec3 cB = (verts[triangles[b].v_indices[0]] + verts[triangles[b].v_indices[1]] + verts[triangles[b].v_indices[2]]) / 3.0f;

            if (x_dist >= y_dist && x_dist >= z_dist) {
                return cA.x < cB.x;
            }
            else if (y_dist >= x_dist && y_dist >= z_dist) {
                return cA.y < cB.y;
            }
            else if (z_dist >= x_dist && z_dist >= y_dist) {
                return cA.z < cB.z;
            }
            else {
                printf("UNREACHABLE\n");
                exit(1);
            }
        }
    );

    int num_leaves = (end - start + 1);
    int log = glm::floor(glm::log2(float(num_leaves))) - 1.0f;
    int partition = glm::pow(2, log);
    int rem = num_leaves - 2 * partition;
    int left_alloc = min(rem, partition);
    int right_alloc = rem - left_alloc;

    int window_size_a = partition + left_alloc;
    int window_size_b = partition + right_alloc;

    int start_a = start;
    int end_a = start + window_size_a - 1;
    int start_b = end_a + 1;
    int end_b = end;

    if ( end_b - start_b + 1 != window_size_b ) {
        printf("Error: end_b - start_b (%d) != right_alloc (%d)\n", end_b - start_b + 1, right_alloc);
    }

    if ( window_size_a < window_size_b ) {
        printf("Window Size A %d was smaller than Window Size B %d\n", window_size_a, window_size_b);
    }

    int left = allocated_length;
    int right = allocated_length + 1;
    allocated_length += 2;

    BoundingBox bbox_a = fill_bvh(left, start_a, end_a, h_bvh, verts, triangles, tri_indices, allocated_length);
    BoundingBox bbox_b = fill_bvh(right, start_b, end_b, h_bvh, verts, triangles, tri_indices, allocated_length);

    glm::vec3 combined_bbox[4] = {
        bbox_a.box_max,
        bbox_a.box_min,
        bbox_b.box_max,
        bbox_b.box_min,
    };

    h_bvh[idx] = BVHNode();
    auto fin_bbox = BoundingBox(combined_bbox, 4);
    h_bvh[idx].make_bvh_node(fin_bbox, left);

    return fin_bbox;
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

__host__ __device__
bool BoundingBox::RayBoxInterection(Ray ray) {
    float tmin = -FLT_MAX;
    float tmax =  FLT_MAX;

    ray.direction = glm::normalize(ray.direction);

    for (int i = 0; i < 3; i++) {
        if (fabs(ray.direction[i]) < 1e-8f) {
            if (ray.origin[i] < box_min[i] || ray.origin[i] > box_max[i]) {
                return false;
            }
        } 
        else {
            float invD = 1.0f / ray.direction[i];
            float t0 = (box_min[i] - ray.origin[i]) * invD;
            float t1 = (box_max[i] - ray.origin[i]) * invD;
            if (t0 > t1) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            tmin = glm::max(tmin, t0);
            tmax = glm::min(tmax, t1);

            if (tmax < tmin) {
                return false;
            }
        }
    }

    // tmin_out = tmin;
    // tmax_out = tmax;
    return true;
}