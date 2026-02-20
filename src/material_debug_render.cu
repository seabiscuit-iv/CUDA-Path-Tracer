#include "material_debug_render.h"

#include "shaders/lambert.h"

__device__ void render_material_debug_mode(
    const Material& material, 
    PathSegment& path, 
    glm::vec3 materialColor, 
    glm::vec3 normal, 
    glm::vec3 normal_map,
    int idx,
    int num_paths,
    int iter,
    int depth,
    thrust::default_random_engine& rng,
    int material_debug_mode
) {
    glm::vec3 debug_color = 
        material_debug_mode == 1 ? materialColor :
        material_debug_mode == 2 ? normal :
        /* DEV_OPTIONS.material_debug_mode == 3 ? */ normal_map;

    if (material.material_type == MaterialType::Emissive && !path.kill) {
        path.color += path.throughput * material.emittance * material.color;
        path.kill = true;
    }
    else {
        Lambert::sampleHemisphere(idx, num_paths, iter, depth, path, rng, normal);
        glm::vec3 lambert = Lambert::shadePathLambert(idx, iter, num_paths, depth, path, material, debug_color, normal, path.sample_dir);
        float pdf = Lambert::PDF(path.sample_dir, normal);
        path.throughput *= lambert / pdf;
    }
}