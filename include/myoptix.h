#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <fmt/format.h>
#include <exception>

#include "sceneStructs.h"
#include "optixshaders/params/optix_triangle_params.cuh"

void init_optix();
OptixDeviceContext get_optix();

#define OPTIX_CHECK(call)                                                      \
do {                                                                           \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                               \
        fprintf(stderr, "OptiX call (%s) failed with code %d\n", #call, res); \
        exit(1);                                                              \
    }                                                                          \
} while(0)

#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x
#define LINE_STR STRINGIFY( __LINE__ )

#define NVRTC_CHECK_ERROR( func )                                                                                           \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if( code != NVRTC_SUCCESS )                                                                                         \
            throw std::runtime_error( fmt::format("ERROR: {} ({}): {}", __FILE__, LINE_STR, std::string( nvrtcGetErrorString( code ) )) ); \
    } while( 0 )

static inline void optixCheckLog( OptixResult  res,
                           const char*  log,
                           size_t       sizeof_log,
                           size_t       sizeof_log_returned,
                           const char*  call,
                           const char*  file,
                           unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        throw std::runtime_error( fmt::format("Optix call '{}' failed: {}: {} \nLog:\n{} {}\n", call, file, line, log, ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" )) );
    }
}

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call,     \
                                __FILE__, __LINE__ );                          \
    } while( false )

#define CUDA_NVRTC_OPTIONS  \
  "-std=c++17", \
  "-arch", \
  "compute_75", \
  "-lineinfo", \
  "-use_fast_math", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64"

void getCuStringFromFile( std::string& cu, const char* filename );
void getInputFromCuString( std::string&                    input,                                  
                                  const char*                     cu_source,
                                  const char*                     name,
                                  const char**                    log_string = nullptr,
                                  const std::vector<const char*>& compiler_options = {CUDA_NVRTC_OPTIONS});

#define ABSOLUTE_INCLUDE_DIRS \
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include", \
  "C:/Users/saahi/Documents/Homework/cis-5650/CUDA-Path-Tracer/external/include", \
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include"
//   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/cccl/cuda/std", \
//   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/cccl" 


void build_optix_accel_structure(
    const std::vector<glm::vec3>& h_verts, 
    const glm::vec3* d_verts, 
    const std::vector<Triangle>& h_triangles, 
    const Triangle* d_triangles,
    OptixTraversableHandle& out_handle,
    CUdeviceptr& out_buffer
); 

void compile_pathtracing_optix_module(OptixModule& out_module, OptixPipelineCompileOptions& out_pipeline_options);

void create_optix_program_groups(
    const OptixModule& module, 
    OptixProgramGroup& out_raygen_prog_group, 
    OptixProgramGroup& out_miss_prog_group, 
    OptixProgramGroup& out_directlight_miss_prog_group,
    OptixProgramGroup& out_hit_prog_group, 
    OptixProgramGroup& out_directlight_prog_group
);

void initialize_optix_pipeline(
    const OptixProgramGroup& raygen_prog_group, 
    const OptixProgramGroup& miss_prog_group, 
    const OptixProgramGroup& directlight_miss_prog_group, 
    const OptixProgramGroup& hitgroup_prog_group, 
    const OptixProgramGroup& directlight_prog_group, 
    const OptixPipelineCompileOptions& pipeline_compile_options,
    OptixPipeline& pipeline
);

void create_optix_sbt(
    OptixShaderBindingTable& sbt, 
    const OptixProgramGroup& raygen_prog_group, 
    const OptixProgramGroup& miss_prog_group, 
    const OptixProgramGroup& directlight_miss_prog_group, 
    const OptixProgramGroup& hitgroup_program_group,
    const OptixProgramGroup& directlight_hitgroup_program_group
);

void create_ias(
    const std::vector<OptixInstance>& instances,
    CUdeviceptr &d_optix_instances,
    OptixTraversableHandle& ias_handle
);

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;