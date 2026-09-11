#pragma once

#include "sceneStructs.h"
#include <vector>
#include "optix.h"
#include "texture.h"
#include <glm/glm.hpp>
#include "tinygltf/tiny_gltf.h"

class Scene
{
private:

#if LOAD_FROM_JSON
    void loadFromJSON(const std::string& jsonName, std::string env_map_path);
#endif

    void loadFromGLTF(const std::string& gltfName, std::string env_map_path);
public:
    Scene(std::string filename, const char* env_map);

    std::vector<Geom> geoms;
    RenderState state;

    std::vector<Material> materials;
    std::vector<std::string> material_names;

    std::vector<glm::vec4> exr_data;
    int exr_width;
    int exr_height;
    
    OptixPipeline optix_pipeline;
    OptixTraversableHandle ias_handle;
    OptixShaderBindingTable optix_sbt;

    // funny light sampling stuff
    float total_emissive_mesh_area;
    std::vector<int> emissive_geoms;
    std::vector<float> emissive_geom_area_prefix;

    // hdri importance sampling stuff
    float total_hdri_emission;
    std::vector<float> hdri_marginal_cdf; // size == exr_height
    std::vector<float> hdri_conditional_cdfs; // size == exr_width * exr_height

    void precompute_emissive_mesh_area();
    void precompute_hdri_emission();
};
