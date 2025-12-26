/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <optix.h>

#include <sutil/vec_math.h>
#include <sutil/helpers.h>

#include "optix_triangle_params.cuh"

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    // const float3 U = params.cam_u;
    // const float3 V = params.cam_v;
    // const float3 W = params.cam_w;
    // const float2 d = 2.0f * make_float2(
    //         static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
    //         static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    //         ) - 1.0f;

    // origin    = params.cam_eye;
    // direction = normalize( d.x * U + d.y * V + W );
    OptixPathSegment& path_segment = params.path_segments[idx.x];

    origin = path_segment.ray.origin;
    direction = path_segment.ray.direction;

    // params.debug_image[path_segment.pixelIndex] = make_float3(1.0, 0.0, 0.0);
    // params.debug_image[path_segment.pixelIndex] = direction;
}


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            RAY_TYPE_COUNT,      // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
    // float3 result;
    // result.x = __uint_as_float( p0 );
    // result.y = __uint_as_float( p1 );
    // result.z = __uint_as_float( p2 );

    // const OptixPathSegment& path_segment = params.path_segments[idx.x];
    // OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];

    // unsigned int object_ID = p0;
    // unsigned int prim_ID = p2;
    
    // if (object_ID == 0xFFFFFFFF) {
    //     params.debug_image[path_segment.pixelIndex].x = 0.0;
    // }  
    // else {
    //     int material_id = params.material_ids[object_ID];
    //     // params.debug_image[path_segment.pixelIndex].x = (float)(material_id) / 10.0f;

    //     float3* vertex_buffer = params.vertex_buffer_locations[object_ID];
    //     OptixTriangle* triangle_buffer = params.triangle_buffer_locations[object_ID];
    //     float3* normal_buffer = params.normal_buffer_locations[object_ID];

    //     OptixTriangle& triangle = triangle_buffer[prim_ID];

    //     float3 A = vertex_buffer[triangle.v_indices[0]];
    //     float3 B = vertex_buffer[triangle.v_indices[1]];
    //     float3 C = vertex_buffer[triangle.v_indices[2]];

    //     float3 edge1 = B - A;
    //     float3 edge2 = C - A;

    //     float3 normal = normalize(cross(edge1, edge2));

    //     normal = normalize(optixTransformNormalFromObjectToWorldSpace())

    //     shadeable_intersection.materialId = material_id;
    //     shadeable_intersection.t = __uint_as_float(p1);

    //     // params.debug_image[path_segment.pixelIndex].x = (float)(shadeable_intersection.t) / 20.0f;
    //     // params.debug_image[path_segment.pixelIndex] = (normal + 1.0) / 2.0f;
    //     params.debug_image[path_segment.pixelIndex] = normal;
    // }

    // // // params.shadeable_intersections[]
    // // OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];
    // // if (object_ID < 0) {
    // //     shadeable_intersection.t = -1.0f;
    // // }
    // // else {
    // //     shadeable_intersection.materialId = params.material_ids[object_ID];
    // // }

    // // params.debug_image[path_segment.pixelIndex] = result;

    // // Record results in our output raster
    // // params.image[idx.y * params.image_width + idx.x] = sutil::make_color( result );
}


extern "C" __global__ void __miss__ms()
{
    const uint3 idx = optixGetLaunchIndex();
    const OptixPathSegment& path_segment = params.path_segments[idx.x];
    OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];

    shadeable_intersection.t = -1.0f;
    // params.debug_image[path_segment.pixelIndex].x = float(800 - path_segment.pixelIndex % 800) / 800.0f;
    // params.debug_image[path_segment.pixelIndex].y = float(800 - path_segment.pixelIndex / 800) / 800.0f;

    // shadeable_intersection.surfaceNormal = make_float3(1.0);
}


extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int object_ID = optixGetInstanceId();
    const unsigned int prim_ID = optixGetPrimitiveIndex();

    const uint3 idx = optixGetLaunchIndex();
    const OptixPathSegment& path_segment = params.path_segments[idx.x];
    OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];

    int material_id = params.material_ids[object_ID];

    float3* vertex_buffer = params.vertex_buffer_locations[object_ID];
    OptixTriangle* triangle_buffer = params.triangle_buffer_locations[object_ID];
    float3* normal_buffer = params.normal_buffer_locations[object_ID];

    OptixTriangle& triangle = triangle_buffer[prim_ID];

    float3 A = vertex_buffer[triangle.v_indices[0]];
    float3 B = vertex_buffer[triangle.v_indices[1]];
    float3 C = vertex_buffer[triangle.v_indices[2]];

    float3 edge1 = B - A;
    float3 edge2 = C - A;

    float3 normal = normalize(cross(edge1, edge2));

    normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

    shadeable_intersection.materialId = material_id;
    shadeable_intersection.t = optixGetRayTmax();

    // params.debug_image[path_segment.pixelIndex].x = (float)(shadeable_intersection.t) / 20.0f;
    // params.debug_image[path_segment.pixelIndex] = (normal + 1.0) / 2.0f;
    shadeable_intersection.surfaceNormal = normal;
}
