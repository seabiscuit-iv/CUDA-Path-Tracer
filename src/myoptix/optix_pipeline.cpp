#include "myoptix/optix_pipeline.h"
#include "myoptix/optix_check.h"
#include "myoptix/optix_context.h"

#include <optix_stubs.h>

// glm must precede optix_stack_size.h
#include <glm/glm.hpp>
#include <optix_stack_size.h>

#include <fmt/format.h>

void initialize_optix_pipeline(
    const OptixProgramGroup& raygen_prog_group, 
    const OptixProgramGroup& miss_prog_group, 
    const OptixProgramGroup& directlight_miss_prog_group, 
    const OptixProgramGroup& hitgroup_prog_group, 
    const OptixProgramGroup& directlight_hitgroup_prog_group, 
    const OptixProgramGroup& envmap_miss_prog_group,
    const OptixProgramGroup& envmap_hit_prog_group,
    const OptixPipelineCompileOptions& pipeline_compile_options,
    OptixPipeline& pipeline
) {
    const uint32_t    max_trace_depth  = 1;
    OptixProgramGroup program_groups[] = { 
        raygen_prog_group, miss_prog_group, 
        directlight_miss_prog_group, 
        hitgroup_prog_group, 
        directlight_hitgroup_prog_group,
        envmap_miss_prog_group,
        envmap_hit_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
    OPTIX_CHECK_LOG( optixPipelineCreate(
                optix,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                LOG, &LOG_SIZE,
                &pipeline
                ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                0,  // maxCCDepth
                                                0,  // maxDCDEpth
                                                &direct_callable_stack_size_from_traversal,
                                                &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            2  // maxTraversableDepth
                                            ) );

    fmt::println("Pipeline Created Successfully");
}
