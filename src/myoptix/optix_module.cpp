#include "myoptix/optix_module.h"
#include "myoptix/optix_check.h"
#include "myoptix/optix_context.h"
#include "myoptix/optix_nvrtc.h"

#include <optix_stubs.h>
#include <string>
#include <fmt/format.h>

#include "config.h"

void compile_pathtracing_optix_module(OptixModule& module, OptixPipelineCompileOptions& pipeline_compile_options) {
    OptixDeviceContext optix = get_optix();

    OptixModuleCompileOptions module_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues      = 1; // occlusion flag for the env map shadow ray
    pipeline_compile_options.numAttributeValues    = 3; // fix later
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params"; // fix later

    std::string shaderfile = "optix_triangle.cu";
    std::string cu, input;

    getCuStringFromFile(cu, shaderfile.c_str());
    getInputFromCuString(input, cu.c_str(), "optix_triangle");

    OPTIX_CHECK_LOG( optixModuleCreate(
        optix,
        &module_compile_options,
        &pipeline_compile_options,
        input.c_str(),
        input.size(),
        LOG, &LOG_SIZE,
        &module
        ) );

}
