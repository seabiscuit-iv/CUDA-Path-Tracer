#include "mesh/bounding_box.h"

#include "sceneStructs.h"

#include <glm/glm.hpp>

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
