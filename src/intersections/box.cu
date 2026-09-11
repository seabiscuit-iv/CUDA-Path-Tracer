#include "intersections.h"
#include "stack.h"

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
