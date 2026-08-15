#include "material_debug_render.h"

__device__ void render_material_debug_mode(
    PathSegment& path, 
    glm::vec3 materialColor, 
    glm::vec3 normal, 
    glm::vec3 normal_map,
    int material_debug_mode
) {
    glm::vec3 debug_color = 
        material_debug_mode == 1 ? materialColor :
        material_debug_mode == 2 ? normal * 0.5f + 0.5f :
        /* material_debug_mode == 3 ? */ normal_map;

    path.color = debug_color;
    path.kill = true;
}
