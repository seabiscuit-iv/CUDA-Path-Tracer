#pragma once

#include <optix.h>

#include <cuda_runtime.h>

#include "optixshaders/params/optix_triangle_params.cuh"

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

void create_optix_sbt(
    OptixShaderBindingTable& sbt, 
    const OptixProgramGroup& raygen_prog_group, 
    const OptixProgramGroup& miss_prog_group, 
    const OptixProgramGroup& directlight_miss_prog_group, 
    const OptixProgramGroup& hitgroup_program_group,
    const OptixProgramGroup& directlight_hitgroup_program_group,
    const OptixProgramGroup& envmap_hit_prog_group,
    const OptixProgramGroup& envmap_miss_prog_group
);
