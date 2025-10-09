#include "intersections.h"
#include "stack.h"
#include "pathtrace.h"

__device__ float boxIntersectionTest(
    const Geom &box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f));
    
    q.inv_direction.x = __frcp_rn(q.direction.x);
    q.inv_direction.y = __frcp_rn(q.direction.y);
    q.inv_direction.z = __frcp_rn(q.direction.z);

    q.sign.x = (q.inv_direction.x < 0.0) ? 1 : 0;
    q.sign.y = (q.inv_direction.y < 0.0) ? 1 : 0;
    q.sign.z = (q.inv_direction.z < 0.0) ? 1 : 0;

    glm::vec3 bounds[2] = { glm::vec3(-0.5), glm::vec3(0.5) }; 
    
    float t_min, t_max;
    int hit_axis = 0; // 0=x, 1=y, 2=z
    
    t_min = (bounds[q.sign.x].x - q.origin.x) * q.inv_direction.x;
    t_max = (bounds[1 - q.sign.x].x - q.origin.x) * q.inv_direction.x;

    float t_ymin = (bounds[q.sign.y].y - q.origin.y) * q.inv_direction.y;
    float t_ymax = (bounds[1 - q.sign.y].y - q.origin.y) * q.inv_direction.y;

    if (t_min > t_ymax || t_ymin > t_max) {
        return -1.0;
    }
    if (t_ymin > t_min) {
        t_min = t_ymin; 
        hit_axis = 1; 
    }
    if (t_ymax < t_max) {
        t_max = t_ymax;
    }

    float t_zmin = (bounds[q.sign.z].z - q.origin.z) * q.inv_direction.z;
    float t_zmax = (bounds[1 - q.sign.z].z - q.origin.z) * q.inv_direction.z;

    if (t_min > t_zmax || t_zmin > t_max) return -1.0;
    if (t_zmin > t_min) { t_min = t_zmin; hit_axis = 2; }

    if (t_max < 0.0f) return -1.0;
    
    if (t_min < 0.0f) {
        t_min = t_max;
        outside = false;
    } else {
        outside = true;
    }

    glm::vec3 N = glm::vec3(0.0f);
    if (hit_axis == 0) N.x = q.sign.x == 1 ? 1.0f : -1.0f;
    if (hit_axis == 1) N.y = q.sign.y == 1 ? 1.0f : -1.0f;
    if (hit_axis == 2) N.z = q.sign.z == 1 ? 1.0f : -1.0f;

    intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, t_min), 1.0f));
    normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(N, 0.0f)));

    return t_min;
}

__host__ __device__ float sphereIntersectionTest(
    Geom &sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float sphereIntersectionTest(
    const Geom &sphere,
    const Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    const float radius = 0.5f;

    // Transform ray into object space
    glm::vec3 ro = glm::vec3(multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f)));
    glm::vec3 rd = glm::vec3(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float b = glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - radius * radius;
    float disc = b * b - c;

    if (disc < 0.0f) return -1.0f;

    float sqrtDisc = sqrtf(disc);
    float t1 = -b - sqrtDisc;
    float t2 = -b + sqrtDisc;

    float t;
    if (t1 > 0.0f) {
        t = t1;
        outside = true;
    } else if (t2 > 0.0f) {
        t = t2;
        outside = false;
    } else {
        return -1.0f;
    }

    // Compute intersection in object space
    glm::vec3 p = ro + t * rd;

    // Transform back to world space
    intersectionPoint = glm::vec3(multiplyMV(sphere.transform, glm::vec4(p, 1.0f)));

    // Compute normal
    normal = glm::vec3(multiplyMV(sphere.invTranspose, glm::vec4(p, 0.0f)));
    normal = glm::normalize(normal);
    if (!outside) normal = -normal;

    // Return distance in world space (can also just return t if ray directions are normalized)
    return glm::length(r.origin - intersectionPoint);
}





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

    int num_verts = mesh.mesh.num_verts;
    glm::vec3* verts = mesh.mesh.d_verts;
    int num_tris = mesh.mesh.num_triangles;
    Triangle* triangles = mesh.mesh.d_triangles;

    int num_normals = mesh.mesh.num_normals;
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

            if ((u < 0 && glm::abs(u) > epsilon) || (u > 1 && glm::abs(u-1) > epsilon)) {
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