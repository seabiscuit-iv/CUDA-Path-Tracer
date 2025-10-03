#include "mesh.h"
#include <stack>
#include "stack.h"
#include "sceneStructs.h"
#include <algorithm>
#include <glm/glm.hpp>
#include <numeric>

#define BIN_COUNT 16


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

    // SAH CALCULATION WITH BINNING
            std::vector<glm::vec3> centroids;

            for(int i = start; i <= end; i++) {
                glm::vec3 centroid = triangles[tri_indices[i]].centroid(verts);
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

            float x_bin_width = (x_dist > 0) ? x_dist / BIN_COUNT : 1.0f;
            float y_bin_width = (y_dist > 0) ? y_dist / BIN_COUNT : 1.0f;
            float z_bin_width = (z_dist > 0) ? z_dist / BIN_COUNT : 1.0f;

            std::vector<glm::ivec3> bin_indices(end-start+1);
            for (int i = 0; i < bin_indices.size(); i++) {
                bin_indices[i].x = glm::clamp( int((centroids[i].x - x_min) / x_bin_width), 0, BIN_COUNT - 1 );
                bin_indices[i].y = glm::clamp( int((centroids[i].y - y_min) / y_bin_width), 0, BIN_COUNT - 1 );
                bin_indices[i].z = glm::clamp( int((centroids[i].z - z_min) / z_bin_width), 0, BIN_COUNT - 1 );
            }

            std::vector<int> x_bin_counts(BIN_COUNT, 0);
            std::vector<int> y_bin_counts(BIN_COUNT, 0);
            std::vector<int> z_bin_counts(BIN_COUNT, 0);
            for (auto t : bin_indices) {
                x_bin_counts[t.x] += 1;
                y_bin_counts[t.y] += 1;
                z_bin_counts[t.z] += 1;
            }

            std::vector<BoundingBox> x_bin_bounds(BIN_COUNT);
            std::vector<BoundingBox> y_bin_bounds(BIN_COUNT);
            std::vector<BoundingBox> z_bin_bounds(BIN_COUNT);
            for(int i = 0; i < (end-start+1); i++) {
                glm::vec3 centroid = centroids[i];
                int bx = bin_indices[i].x;
                int by = bin_indices[i].y;
                int bz = bin_indices[i].z;

                x_bin_bounds[bx].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[0]]);
                x_bin_bounds[bx].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[1]]);
                x_bin_bounds[bx].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[2]]);

                y_bin_bounds[by].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[0]]);
                y_bin_bounds[by].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[1]]);
                y_bin_bounds[by].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[2]]);

                z_bin_bounds[bz].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[0]]);
                z_bin_bounds[bz].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[1]]);
                z_bin_bounds[bz].expandBoxByVec3(verts[triangles[tri_indices[i + start]].v_indices[2]]);
            }

            auto expand = [](BoundingBox a, const BoundingBox &b) {
                a.expandBox(b);
                return a;
            };

            std::vector<int> x_leftCount(BIN_COUNT);
            std::inclusive_scan(x_bin_counts.begin(), x_bin_counts.end(),
                                x_leftCount.begin(),
                                std::plus<int>());
            std::vector<BoundingBox> x_leftBounds(BIN_COUNT);
            std::inclusive_scan(x_bin_bounds.begin(), x_bin_bounds.end(),
                                x_leftBounds.begin(),
                                expand );

            std::vector<int> y_leftCount(BIN_COUNT);
            std::inclusive_scan(y_bin_counts.begin(), y_bin_counts.end(),
                                y_leftCount.begin(),
                                std::plus<int>());
            std::vector<BoundingBox> y_leftBounds(BIN_COUNT);
            std::inclusive_scan(y_bin_bounds.begin(), y_bin_bounds.end(),
                                y_leftBounds.begin(),
                                expand );

            std::vector<int> z_leftCount(BIN_COUNT);
            std::inclusive_scan(z_bin_counts.begin(), z_bin_counts.end(),
                                z_leftCount.begin(),
                                std::plus<int>());
            std::vector<BoundingBox> z_leftBounds(BIN_COUNT);
            std::inclusive_scan(z_bin_bounds.begin(), z_bin_bounds.end(),
                                z_leftBounds.begin(),
                                expand );



            std::vector<int> x_rightCount(BIN_COUNT);
            std::inclusive_scan(x_bin_counts.rbegin(), x_bin_counts.rend(),
                                x_rightCount.rbegin(),
                                std::plus<int>());
            std::vector<BoundingBox> x_rightBounds(BIN_COUNT);
            std::inclusive_scan(x_bin_bounds.rbegin(), x_bin_bounds.rend(),
                                x_rightBounds.rbegin(),
                                expand );

            std::vector<int> y_rightCount(BIN_COUNT);
            std::inclusive_scan(y_bin_counts.rbegin(), y_bin_counts.rend(),
                                y_rightCount.rbegin(),
                                std::plus<int>());
            std::vector<BoundingBox> y_rightBounds(BIN_COUNT);
            std::inclusive_scan(y_bin_bounds.rbegin(), y_bin_bounds.rend(),
                                y_rightBounds.rbegin(),
                                expand );

            std::vector<int> z_rightCount(BIN_COUNT);
            std::inclusive_scan(z_bin_counts.rbegin(), z_bin_counts.rend(),
                                z_rightCount.rbegin(),
                                std::plus<int>());
            std::vector<BoundingBox> z_rightBounds(BIN_COUNT);
            std::inclusive_scan(z_bin_bounds.rbegin(), z_bin_bounds.rend(),
                                z_rightBounds.rbegin(),
                                expand );

    
            float x_min_cost = FLT_MAX;
            int best_x_bin = -1;
            // x SAH calculation
            for(int i = 0; i <= BIN_COUNT-2; i++) {
                int x_lc  = x_leftCount[i];
                int x_rc = x_rightCount[i + 1];

                BoundingBox x_lbb  = x_leftBounds[i];
                BoundingBox x_rbb = x_rightBounds[i + 1];

                int tc = end - start + 1;

                float cost = 1 + 
                    (float(x_lc) / float(tc)) * x_lbb.surfaceArea() +
                    (float(x_rc) / float(tc)) * x_rbb.surfaceArea();

                if (cost < x_min_cost) {
                    x_min_cost = cost;
                    best_x_bin = i;
                }
            }

            float y_min_cost = FLT_MAX;
            int best_y_bin = -1;
            // y SAH calculation
            for(int i = 0; i <= BIN_COUNT-2; i++) {
                int y_lc  = y_leftCount[i];
                int y_rc = y_rightCount[i + 1];

                BoundingBox y_lbb  = y_leftBounds[i];
                BoundingBox y_rbb = y_rightBounds[i + 1];

                int tc = end - start + 1;

                float cost = 1 + 
                    (float(y_lc) / float(tc)) * y_lbb.surfaceArea() +
                    (float(y_rc) / float(tc)) * y_rbb.surfaceArea();

                if (cost < y_min_cost) {
                    y_min_cost = cost;
                    best_y_bin = i;
                }
            }


            float z_min_cost = FLT_MAX;
            int best_z_bin = -1;
            // y SAH calculation
            for(int i = 0; i <= BIN_COUNT-2; i++) {
                int z_lc  = z_leftCount[i];
                int z_rc = z_rightCount[i + 1];

                BoundingBox z_lbb  = z_leftBounds[i];
                BoundingBox z_rbb = z_rightBounds[i + 1];

                int tc = end - start + 1;

                float cost = 1 + 
                    (float(z_lc) / float(tc)) * z_lbb.surfaceArea() +
                    (float(z_rc) / float(tc)) * z_rbb.surfaceArea();

                if (cost < z_min_cost) {
                    z_min_cost = cost;
                    best_z_bin = i;
                }
            }

            int best_axis = -1;
            int best_bin = -1;
            if (x_min_cost <= y_min_cost && x_min_cost <= z_min_cost) {
                best_axis = 0;
                best_bin = best_x_bin;
            }
            else if (y_min_cost <= x_min_cost && y_min_cost <= z_min_cost) {
                best_axis = 1;
                best_bin = best_y_bin;
            }
            else if (z_min_cost <= x_min_cost && z_min_cost <= y_min_cost) {
                best_axis = 2;
                best_bin = best_z_bin;
            }
            else {
                printf("UNREACHABLE\n");
                exit(1);
            }

            std::string ax;

            if (best_axis == 0) {
                ax = "X";
            }
            else if (best_axis == 1) {
                ax = "Y";
            }
            else {
                ax = "Z";
            }

            auto pivot = std::partition(
                tri_indices.begin() + start,
                tri_indices.begin() + end + 1,
                [&](int idx) {
                glm::vec3 c = triangles[idx].centroid(verts);
                    int bin;
                    if (best_axis == 0) {
                        bin = glm::clamp( int((c.x - x_min) / x_bin_width), 0, BIN_COUNT - 1 );
                    }
                    else if (best_axis == 1) {
                        bin = glm::clamp( int((c.y - y_min) / y_bin_width), 0, BIN_COUNT - 1 );
                    }
                    else {
                        bin = glm::clamp( int((c.z - z_min) / z_bin_width), 0, BIN_COUNT - 1 );
                    }

                    return bin <= best_bin;
                }
            );

            int mid = std::distance(tri_indices.begin(), pivot);

            float lower_split_ratio = float(mid - start) / float(end - start + 1);
            float upper_split_ratio = float(end - mid + 1) / float(end - start + 1);

            if (lower_split_ratio < 0.15f || upper_split_ratio < 0.15f) {
                // printf("Start %d, End %d, Mid %d\n", start, end, mid);
                int new_mid = (start + end + 1) / 2;

                std::nth_element(
                    tri_indices.begin() + start, 
                    tri_indices.begin() + new_mid, 
                    tri_indices.begin() + end + 1,
                    [&](int idx_a, int idx_b) {
                        glm::vec3 c_a = triangles[idx_a].centroid(verts);
                        glm::vec3 c_b = triangles[idx_b].centroid(verts);
                        return c_a[best_axis] < c_b[best_axis];
                    }
                );

                mid = new_mid;
            }

            // printf("Start: %d, End: %d \nAxis: %s \nMid: %d\n\n", start, end, ax.c_str(), mid);
    // END SAH CALCULATION

    int left = allocated_length;
    int right = allocated_length + 1;
    allocated_length += 2;

    BoundingBox bbox_a = fill_bvh(left, start, mid-1, h_bvh, verts, triangles, tri_indices, allocated_length);
    BoundingBox bbox_b = fill_bvh(right, mid, end, h_bvh, verts, triangles, tri_indices, allocated_length);

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
bool BoundingBox::RayBoxInterection(const Ray& ray, float& t_hit_min) {
    glm::vec3 bounds[2] = { box_min, box_max }; 
    
    float t_min, t_max;
    
    t_min = (bounds[ray.sign.x].x - ray.origin.x) * ray.inv_direction.x;
    t_max = (bounds[1 - ray.sign.x].x - ray.origin.x) * ray.inv_direction.x;

    float t_ymin = (bounds[ray.sign.y].y - ray.origin.y) * ray.inv_direction.y;
    float t_ymax = (bounds[1 - ray.sign.y].y - ray.origin.y) * ray.inv_direction.y;

    if (t_min > t_ymax || t_ymin > t_max) { 
        return false;
    }

    t_min = glm::max(t_min, t_ymin);
    t_max = glm::min(t_max, t_ymax);

    float t_zmin = (bounds[ray.sign.z].z - ray.origin.z) * ray.inv_direction.z;
    float t_zmax = (bounds[1 - ray.sign.z].z - ray.origin.z) * ray.inv_direction.z;

    if (t_min > t_zmax || t_zmin > t_max) {
        return false;
    }

    t_min = glm::max(t_min, t_zmin);
    
    if (t_max < 0.0f) {
        return false;
    } 
    
    t_hit_min = t_min;
    return t_min < t_max;
}


