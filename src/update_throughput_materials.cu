#include "update_throughput_materials.h"

#include "shaders/lambert.h"
#include "shaders/cook_torrance.h"
#include "shaders/glass.h"
#include "shaders/specular.h"

__device__ void update_throughput_materials (
    const Material& material,
    PathSegment& path,
    int idx,
    int num_paths,
    int iter,
    int depth,
    thrust::default_random_engine& rng,
    glm::vec3 materialColor,
    glm::vec3 normal,
    bool is_specular
) {
    if (material.material_type == MaterialType::Diffuse) {
        glm::vec3 lambert = Lambert::shadePathLambert(idx, iter, num_paths, depth, path, material, materialColor, normal, path.sample_dir);
        float pdf = Lambert::PDF(path.sample_dir, normal);
        path.throughput *= lambert / pdf;
        path.last_pdf = pdf;
    } 
    else if (material.material_type == MaterialType::Specular) {
        PerfectSpecular::shadePathSpecular(path, material, materialColor);
        path.last_pdf = 1.0;
    }
    else if (material.material_type == MaterialType::Microfacet) {
        glm::vec3 cook_torrance = CookTorrance::shadePathCookTorrance(path, material, materialColor, normal, path.sample_dir);
        float pdf = CookTorrance::PDF(material, -path.ray.direction, path.sample_dir, normal, material.roughness, materialColor);
        path.throughput *= cook_torrance / pdf;
        path.last_pdf = pdf;
    }
    else if (material.material_type == MaterialType::Glass) {
        TransmissiveGlass::shadePathGlass(path, material, materialColor);
        path.last_pdf = 1.0;
    }

    path.last_bounce_was_specular = is_specular;

    if (depth >= RUSSIAN_ROULETTE_MIN_DEPTH) {
        float max_throughput = glm::max(path.throughput.x, glm::max(path.throughput.y, path.throughput.z));
        float survival = glm::clamp(max_throughput, 0.0f, 1.0f);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) >= survival) {
            path.kill = true;
        }
        else {
            path.throughput /= survival;
        }
    }
}