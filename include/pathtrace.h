#pragma once

#include "scene.h"
#include "utilities.h"
#include "config.h"

#define BLOCK_SIZE_1D 128

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);


struct PathTracerOptions {
    bool debug_bvh = false;
    bool material_debug_mode = false;
    int color_mode = 2;
    float envmap_intensity = 1.0f;
    bool direct_light_sampling = false;
    bool environment_map_importance_sampling = false;

    static PathTracerOptions* Get() {
        static PathTracerOptions instance;
        return &instance;
    }

    // Prevent copying and assignment
    PathTracerOptions(const PathTracerOptions&) = delete;
    PathTracerOptions& operator=(const PathTracerOptions&) = delete;
    PathTracerOptions() = default;
};