#pragma once

struct PathTracerOptions {
    bool debug_bvh = false;
    int material_debug_mode = 0;
    int color_mode = 2;
    float envmap_intensity = 1.0f;
    bool direct_light_sampling = false;
    bool environment_map_importance_sampling = false;
    int selected_material = 0;

    static PathTracerOptions* Get() {
        static PathTracerOptions instance;
        return &instance;
    }

    // Prevent copying and assignment
    PathTracerOptions(const PathTracerOptions&) = delete;
    PathTracerOptions& operator=(const PathTracerOptions&) = delete;
    PathTracerOptions() = default;
};
