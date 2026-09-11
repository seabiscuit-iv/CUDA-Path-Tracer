#include "intersections.h"
#include "stack.h"

#define USE_NORMAL_BUFFERS 1

__device__ float meshIntersectionTest(    
    const Geom &mesh,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside ) 
{
    Ray r_ws = r;   

    r.origin = glm::vec3(mesh.inverseTransform * glm::vec4(r.origin, 1.0f));
    r.direction = glm::vec3(mesh.inverseTransform * glm::vec4(r.direction, 0.0f));

    r.inv_direction.x = __frcp_rn(r.direction.x);
    r.inv_direction.y = __frcp_rn(r.direction.y);
    r.inv_direction.z = __frcp_rn(r.direction.z);

    r.sign.x = (r.inv_direction.x < 0.0f) ? 1 : 0;
    r.sign.y = (r.inv_direction.y < 0.0f) ? 1 : 0;
    r.sign.z = (r.inv_direction.z < 0.0f) ? 1 : 0;

    // at this point, r is now in object space

    BVHNode* bvh = mesh.mesh.bvh.dev_bvh;
    
    Stack dfs_stack;
    dfs_stack.init();
    dfs_stack.push(0);

    float epsilon = (float)(1.1920929E-7F);

    // int num_verts = mesh.mesh.num_verts;
    glm::vec3* verts = mesh.mesh.d_verts;
    // int num_tris = mesh.mesh.num_triangles;
    Triangle* triangles = mesh.mesh.d_triangles;

    // int num_normals = mesh.mesh.num_normals;
    glm::vec3* normals = mesh.mesh.d_normals;

    float min_t = -1.0f;
    glm::vec3 min_isect_point;
    glm::vec3 min_normal;

    while (!dfs_stack.isEmpty()) {
        int bvh_idx = dfs_stack.pop();

        if (!bvh[bvh_idx].isLeaf) {
            int left = bvh[bvh_idx].left_child;
            int right = bvh[bvh_idx].left_child + 1;

            float tmin_left, tmin_right;
            bool hitLeft = bvh[left].isLeaf || bvh[left].box.RayBoxInterection(r, tmin_left);
            bool hitRight = bvh[right].isLeaf || bvh[right].box.RayBoxInterection(r, tmin_right);

            if (hitLeft && tmin_left > min_t && min_t >= 0.0f) {
                hitLeft = false;
            }
            if (hitRight && tmin_right > min_t && min_t >= 0.0f) {
                hitRight = false;
            }

            if (hitLeft && hitRight) {
                dfs_stack.push(tmin_left <= tmin_right ? right : left);
                dfs_stack.push(tmin_left <= tmin_right ? left : right);
            }
            else if (hitLeft) {
                dfs_stack.push(left);
            }
            else if (hitRight) {
                dfs_stack.push(right);
            }
        } else {
            int tri = bvh[bvh_idx].tri_index;

            glm::vec3 a = verts[triangles[tri].v_indices[0]];

            glm::vec3 edge1 = verts[triangles[tri].v_indices[1]] - a;
            glm::vec3 edge2 = verts[triangles[tri].v_indices[2]] - a;

            glm::vec3 cross = glm::cross(r.direction, edge2);
            float det = glm::dot(edge1, cross);

            if (det > -epsilon && det < epsilon) {
                continue;
            }

            float inv_det = __frcp_rn(det);
            a = r.origin - a;
            float u = inv_det * glm::dot(a, cross);

            float u_min = -epsilon;
            float u_max = 1.0f + epsilon;

            if (u < u_min || u > u_max) {
                continue;
            }

            cross = glm::cross(a, edge1);
            float v = inv_det * glm::dot(r.direction, cross);

            if ((v < 0 && glm::abs(v) > epsilon) || (u + v > 1 && glm::abs(u + v - 1) > epsilon)) {
                continue;
            }

            float t = inv_det * glm::dot(edge2,  cross);

            if (t > epsilon) {
                cross = r.origin + t * r.direction;
            } else {
                continue;
            }

            if (min_t < 0 || t < min_t ) {
                min_t = t;
                min_isect_point = cross;

                #if USE_NORMAL_BUFFERS
                    if (mesh.mesh.has_normal_buffers) {
                        // CALCULATE NORMALS FROM HERE
                        glm::vec3 n0 = normals[triangles[tri].n_indices[0]];
                        glm::vec3 n1 = normals[triangles[tri].n_indices[1]];
                        glm::vec3 n2 = normals[triangles[tri].n_indices[2]];

                        glm::vec3 barycentrics(u, v, (1.0f - u - v));
                        glm::vec3 normal = glm::normalize(barycentrics.x * n1 + barycentrics.y * n2 + barycentrics.z * n0);
                        min_normal = normal;
                    } 
                    else 
                #endif
                {
                    min_normal = glm::normalize(glm::cross(edge1, edge2));
                }
            }
        }
    }

    intersectionPoint = r_ws.origin + r_ws.direction * min_t;
    normal = glm::normalize(glm::vec3(mesh.invTranspose * glm::vec4(min_normal, 0.0f)));
    if (glm::dot(normal, r_ws.direction) > 0.0f) {
        normal = -normal;
    }
    outside = glm::dot(normal, r_ws.direction) < 0.0f;

    return min_t;
}
