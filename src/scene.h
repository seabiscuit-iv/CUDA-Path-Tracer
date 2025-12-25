#pragma once

#include "sceneStructs.h"
#include <vector>
#include "optix.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    
    OptixPipeline optix_pipeline;
    OptixTraversableHandle ias_handle;
    OptixShaderBindingTable optix_sbt;
};
