#include "myoptix/optix_ias.h"
#include "myoptix/optix_check.h"
#include "myoptix/optix_context.h"

#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

void create_ias(
    const std::vector<OptixInstance>& optix_instances,
    CUdeviceptr &d_optix_instances,
    OptixTraversableHandle& ias_handle
) {
    size_t size_optix_instances = sizeof(OptixInstance) * optix_instances.size();
    cudaMalloc(reinterpret_cast<void**>(&d_optix_instances), size_optix_instances);
    cudaMemcpy(
        reinterpret_cast<void*>(d_optix_instances),
        optix_instances.data(),
        size_optix_instances,
        cudaMemcpyHostToDevice
    );

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = d_optix_instances;
    build_input.instanceArray.numInstances = static_cast<unsigned int>(optix_instances.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    optixAccelComputeMemoryUsage(
        get_optix(),
        &accel_options,
        &build_input,
        1,
        &ias_buffer_sizes
    );

    CUdeviceptr d_temp;
    CUdeviceptr d_ias;

    cudaMalloc(reinterpret_cast<void**>(&d_temp), ias_buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_ias), ias_buffer_sizes.outputSizeInBytes);
    
    optixAccelBuild(
        get_optix(),
        0,
        &accel_options,
        &build_input,
        1,
        d_temp,
        ias_buffer_sizes.tempSizeInBytes,
        d_ias,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle,
        nullptr,
        0
    );

    fmt::println("IAS Building Complete");
}
