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

    OptixPathSegment& path_segment = params.path_segments[idx.x];

    origin = path_segment.ray.origin;
    direction = path_segment.ray.direction;
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );

    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE,
        0,
        RAY_TYPE_COUNT,
        0
    );
}


extern "C" __global__ void __miss__ms()
{
    const uint3 idx = optixGetLaunchIndex();
    const OptixPathSegment& path_segment = params.path_segments[idx.x];
    OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];

    shadeable_intersection.t = -1.0f;
}


extern "C" __global__ void __closesthit__ch()
{
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
    float2* uv_buffer = params.uv_buffer_locations[object_ID];

    OptixTriangle& triangle = triangle_buffer[prim_ID];

    float3 A = vertex_buffer[triangle.v_indices[0]];
    float3 B = vertex_buffer[triangle.v_indices[1]];
    float3 C = vertex_buffer[triangle.v_indices[2]];

    float3 edge1 = B - A;
    float3 edge2 = C - A;

    float3 normal = normalize(cross(edge1, edge2));

    if (normal_buffer != nullptr) {
        A = normal_buffer[triangle.n_indices[0]];
        B = normal_buffer[triangle.n_indices[1]];
        C = normal_buffer[triangle.n_indices[2]];

        float bA = 1.0f - barycentrics.x - barycentrics.y;

        normal = normalize(bA * A + barycentrics.x * B + barycentrics.y * C);
    }

    float2 uv;
    if (uv_buffer != nullptr) {
        float2 A = uv_buffer[triangle.uv_indices[0]];
        float2 B = uv_buffer[triangle.uv_indices[1]];
        float2 C = uv_buffer[triangle.uv_indices[2]];

        float bA = 1.0f - barycentrics.x - barycentrics.y;

        uv = bA * A + barycentrics.x * B + barycentrics.y * C;
    }

    normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

    shadeable_intersection.materialId = material_id;
    shadeable_intersection.t = optixGetRayTmax();
    shadeable_intersection.u = uv.x;
    shadeable_intersection.v = uv.y;

    shadeable_intersection.surfaceNormal = normal;
}
