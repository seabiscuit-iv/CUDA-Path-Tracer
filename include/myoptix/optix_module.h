#pragma once

#include <optix.h>

void compile_pathtracing_optix_module(OptixModule& out_module, OptixPipelineCompileOptions& out_pipeline_options);
