#include "myoptix/optix_sbt.h"
#include "myoptix/optix_check.h"

#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

void create_optix_sbt(
    OptixShaderBindingTable& sbt, 
    const OptixProgramGroup& raygen_prog_group, 
    const OptixProgramGroup& miss_prog_group, 
    const OptixProgramGroup& directlight_miss_prog_group, 
    const OptixProgramGroup& hitgroup_prog_group,
    const OptixProgramGroup& directlight_hitgroup_prog_group,
    const OptixProgramGroup& envmap_hit_prog_group,
    const OptixProgramGroup& envmap_miss_prog_group
) {
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size );
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
    cudaMemcpy(
        reinterpret_cast<void*>( raygen_record ),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    );

    CUdeviceptr miss_record;
    size_t      miss_record_size = 3 * sizeof( MissSbtRecord );
    cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size );
    MissSbtRecord ms_sbt[3];
    ms_sbt[0].data = { 0.3f, 0.1f, 0.2f };
    ms_sbt[1].data = { 0.3f, 0.1f, 0.2f };
    ms_sbt[2].data = { 0.3f, 0.1f, 0.2f };
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( directlight_miss_prog_group, &ms_sbt[1] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( envmap_miss_prog_group, &ms_sbt[2] ) );
    cudaMemcpy(
        reinterpret_cast<void*>( miss_record ),
        ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    );

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = 3 * sizeof( HitGroupSbtRecord );
    cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size );
    HitGroupSbtRecord hg_sbt[3];
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( directlight_hitgroup_prog_group, &hg_sbt[1] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( envmap_hit_prog_group, &hg_sbt[2] ) );
    cudaMemcpy(
        reinterpret_cast<void*>( hitgroup_record ),
        hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    );

    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 3;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = 3;

    fmt::println("Optix SBT Creation Complete");
}
