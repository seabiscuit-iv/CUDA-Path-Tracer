#pragma once

#include "sceneStructs.h"
#include <vector>
#include "optix.h"
#include <glm/glm.hpp>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    
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