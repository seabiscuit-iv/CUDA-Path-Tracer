#include "myoptix/optix_accel.h"
#include "myoptix/optix_check.h"
#include "myoptix/optix_context.h"

#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

void build_optix_accel_structure(
    const std::vector<glm::vec3>& h_verts, 
    const glm::vec3* d_verts, 
    const std::vector<Triangle>& h_triangles, 
    const Triangle* d_triangles,
    OptixTraversableHandle& as_handle,
    CUdeviceptr& d_as_output_buffer
)  {
    OptixDeviceContext optix = get_optix();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    CUdeviceptr d_verts_cuptr = CUdeviceptr(d_verts);

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( h_verts.size() );
    triangle_input.triangleArray.vertexBuffers = &d_verts_cuptr;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
    triangle_input.triangleArray.flags         = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;
    triangle_input.triangleArray.indexFormat   = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.numIndexTriplets = h_triangles.size();
    triangle_input.triangleArray.indexBuffer = CUdeviceptr(d_triangles);
    triangle_input.triangleArray.indexStrideInBytes = sizeof(Triangle);

    OptixAccelBufferSizes as_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        optix,
        &accel_options,
        &triangle_input,
        1,
        &as_buffer_sizes
        ) );
    // fmt::println("Accel Structure Buffer Size: {} bytes", as_buffer_sizes.outputSizeInBytes);

    CUdeviceptr d_as_temp_buffer;
    cudaMalloc(
                reinterpret_cast<void**>( &d_as_temp_buffer ),
                as_buffer_sizes.tempSizeInBytes
                );
    cudaMalloc(
                reinterpret_cast<void**>( &d_as_output_buffer ),
                as_buffer_sizes.outputSizeInBytes
                );

    CUdeviceptr d_compacted_size;
    cudaMalloc( reinterpret_cast<void**>( &d_compacted_size ), sizeof( size_t ) );

    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = d_compacted_size;

    OPTIX_CHECK( optixAccelBuild(
                optix,
                0,                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                  // num build inputs
                d_as_temp_buffer,
                as_buffer_sizes.tempSizeInBytes,
                d_as_output_buffer,
                as_buffer_sizes.outputSizeInBytes,
                &as_handle,
                &emit_property,     // emitted property list
                1                   // num emitted properties
                ) );

    cudaDeviceSynchronize();
    cudaFree( reinterpret_cast<void*>( d_as_temp_buffer ) );

    size_t compacted_size = 0;
    cudaMemcpy( &compacted_size, reinterpret_cast<void*>( d_compacted_size ), sizeof( size_t ), cudaMemcpyDeviceToHost );
    cudaFree( reinterpret_cast<void*>( d_compacted_size ) );

    if ( compacted_size < as_buffer_sizes.outputSizeInBytes )
    {
        CUdeviceptr d_compacted_buffer;
        cudaMalloc( reinterpret_cast<void**>( &d_compacted_buffer ), compacted_size );

        OPTIX_CHECK( optixAccelCompact(
                    optix,
                    0,              // CUDA stream
                    as_handle,
                    d_compacted_buffer,
                    compacted_size,
                    &as_handle
                    ) );

        cudaDeviceSynchronize();

        cudaFree( reinterpret_cast<void*>( d_as_output_buffer ) );
        d_as_output_buffer = d_compacted_buffer;
    }

    cudaDeviceSynchronize();

}
