#pragma once

#include <optix.h>

void create_optix_program_groups(
    const OptixModule& module, 
    OptixProgramGroup& out_raygen_prog_group, 
    OptixProgramGroup& out_miss_prog_group, 
    OptixProgramGroup& out_directlight_miss_prog_group,
    OptixProgramGroup& out_hit_prog_group, 
    OptixProgramGroup& out_directlight_prog_group,
    OptixProgramGroup& envmap_miss_prog_group,
    OptixProgramGroup& envmap_hit_prog_group
);
