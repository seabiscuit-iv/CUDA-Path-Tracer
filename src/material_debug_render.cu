#include "material_debug_render.h"

__device__ void render_material_debug_mode(
    const Material& material,
    PathSegment& path, 
    glm::vec3 materialColor, 
    glm::vec3 normal, 
    glm::vec3 normal_map,
    int material_debug_mode
) {
    glm::vec3 debug_color(0.0f);

    switch (material_debug_mode) {
        case 1: debug_color = materialColor;               break;
        case 2: debug_color = normal * 0.5f + 0.5f;        break;
        case 3: debug_color = normal_map;                  break;
        case 4: debug_color = glm::vec3(material.roughness); break;
        case 5: debug_color = glm::vec3(material.metallic);  break;
    }

    path.color = debug_color;
    path.kill = true;
}
