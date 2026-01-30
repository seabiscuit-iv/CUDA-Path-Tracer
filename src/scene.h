#pragma once

#include "sceneStructs.h"
#include <vector>
#include "optix.h"
#include <glm/glm.hpp>
#include "tinygltf/tiny_gltf.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, std::string env_map_path);
    void loadFromGLTF(const std::string& gltfName, std::string env_map_path);
public:
    Scene(std::string filename, const char* env_map);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<glm::vec4> exr_data;
    int exr_width;
    int exr_height;
    
    OptixPipeline optix_pipeline;
    OptixTraversableHandle ias_handle;
    OptixShaderBindingTable optix_sbt;
};


static const std::vector<glm::vec3> CUBE_VERTICES = {
    glm::vec3{-0.5f,-0.5f,-0.5f}, glm::vec3{ 0.5f,-0.5f,-0.5f},
    glm::vec3{ 0.5f, 0.5f,-0.5f}, glm::vec3{-0.5f, 0.5f,-0.5f},
    glm::vec3{-0.5f,-0.5f, 0.5f}, glm::vec3{ 0.5f,-0.5f, 0.5f},
    glm::vec3{ 0.5f, 0.5f, 0.5f}, glm::vec3{-0.5f, 0.5f, 0.5f},
};

static const std::vector<int> CUBE_INDICES = {
    0, 2, 1,
    0, 3, 2,
    4, 5, 6,
    4, 6, 7,
    0, 1, 5,
    0, 5, 4,
    3, 6, 2,
    3, 7, 6,
    0, 7, 3,
    0, 4, 7,
    1, 6, 5,
    1, 2, 6
};

static const std::vector<glm::vec3> CUBE_NORMALS = {
    glm::vec3{0.0, 0.0, 1.0}, glm::vec3{0.0, 0.0, -1.0},
    glm::vec3{0.0, 1.0, 0.0}, glm::vec3{0.0, -1.0, 0.0},
    glm::vec3{1.0, 0.0, 0.0}, glm::vec3{-1.0, 0.0, 0.0}
};

static const std::vector<int> CUBE_NORMAL_INDICES = {
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2,
    5, 5, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4
};

inline bool isGlass(const tinygltf::Material& mat) {
    // 1. Explicit Transmission (Modern standard)
    if (mat.extensions.count("KHR_materials_transmission")) return true;

    // 2. Volume (If it has internal absorption/thickness, it's glass)
    if (mat.extensions.count("KHR_materials_volume")) return true;

    // 3. IOR check (If IOR is set and not 1.0, it's likely meant to be refractive)
    if (mat.extensions.count("KHR_materials_ior")) {
        auto it = mat.extensions.find("KHR_materials_ior");
        if (it->second.Has("ior")) {
            double ior = it->second.Get("ior").GetNumberAsDouble();
            if (std::abs(ior - 1.0) > 0.01) return true;
        }
    }

    // 4. Check Alpha in PBR block
    if (mat.alphaMode == "BLEND" || mat.alphaMode == "MASK") {
         if (mat.pbrMetallicRoughness.baseColorFactor[3] < 0.99f) return true;
    }

    // 5. Specular Glossiness Workflow Fallback
    if (mat.extensions.count("KHR_materials_pbrSpecularGlossiness")) {
        auto it = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
        if (it->second.Has("diffuseFactor")) {
            auto factor = it->second.Get("diffuseFactor");
            // Check the 4th element (alpha) of the diffuse array
            if (factor.IsArray() && factor.ArrayLen() == 4) {
                if (factor.Get(3).GetNumberAsDouble() < 0.99) return true;
            }
        }
    }

    return false;
}