#pragma once

#include <optix.h>

void initialize_optix_pipeline(
    const OptixProgramGroup& raygen_prog_group, 
    const OptixProgramGroup& miss_prog_group, 
    const OptixProgramGroup& directlight_miss_prog_group, 
    const OptixProgramGroup& hitgroup_prog_group, 
    const OptixProgramGroup& directlight_prog_group, 
    const OptixProgramGroup& envmap_miss_prog_group,
    const OptixProgramGroup& envmap_hit_prog_group,
    const OptixPipelineCompileOptions& pipeline_compile_options,
    OptixPipeline& pipeline
);
