#include "intersections.h"
#include "stack.h"

__host__ __device__ float boxIntersectionTest(
    Geom &box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
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


#define USE_NORMAL_BUFFERS 1


__host__ __device__ float meshIntersectionTest(    
    Geom &mesh,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside ) 
{
    if (mesh.type != GeomType::MESH) {
        printf("meshIntersectionTest called on non-mesh");
    }

    Ray r_ws = r;   

    r.origin = glm::vec3(mesh.inverseTransform * glm::vec4(r.origin, 1.0f));
    r.direction = glm::vec3(mesh.inverseTransform * glm::vec4(r.direction, 0.0f));

    // at this point, r is now in object space

    BVHNode* bvh = mesh.mesh.bvh.dev_bvh;
    
    Stack dfs_stack;
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
            if(bvh[bvh_idx].box.RayBoxInterection(r)) {
                dfs_stack.push(bvh[bvh_idx].left_child);
                dfs_stack.push(bvh[bvh_idx].left_child + 1);
            }
        } else {
            int tri = bvh[bvh_idx].tri_index;

            glm::vec3 a = verts[triangles[tri].v_indices[0]];
            glm::vec3 b = verts[triangles[tri].v_indices[1]];
            glm::vec3 c = verts[triangles[tri].v_indices[2]];

            glm::vec3 edge1 = b - a;
            glm::vec3 edge2 = c - a;

            glm::vec3 r_cross_e2 = glm::cross(r.direction, edge2);
            float det = glm::dot(edge1, r_cross_e2);

            if (det > -epsilon && det < epsilon) {
                continue;
            }

            float inv_det = 1.0 / det;
            glm::vec3 s = r.origin - a;
            float u = inv_det * glm::dot(s, r_cross_e2);

            if ((u < 0 && glm::abs(u) > epsilon) || (u > 1 && glm::abs(u-1) > epsilon)) {
                continue;
            }

            glm::vec3 s_cross_e1 = glm::cross(s, edge1);
            float v = inv_det * glm::dot(r.direction, s_cross_e1);

            if ((v < 0 && glm::abs(v) > epsilon) || (u + v > 1 && glm::abs(u + v - 1) > epsilon)) {
                continue;
            }

            float t = inv_det * glm::dot(edge2,  s_cross_e1);

            glm::vec3 pos;
            if (t > epsilon) {
                pos = r.origin + t * r.direction;
            } else {
                continue;
            }

            if ( min_t < 0 || t < min_t ) {
                min_t = t;
                min_isect_point = pos;

                #if USE_NORMAL_BUFFERS
                    if (mesh.mesh.has_normal_buffers) {
                        // CALCULATE NORMALS FROM HERE
                        glm::vec3 n0 = normals[triangles[tri].v_indices[0]];
                        glm::vec3 n1 = normals[triangles[tri].v_indices[1]];
                        glm::vec3 n2 = normals[triangles[tri].v_indices[2]];

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