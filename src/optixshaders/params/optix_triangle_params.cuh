#pragma once

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

// struct Params {
//     float3 cam_eye;
//     float4 InvViewProj[4];
//     OptixTraversableHandle as_handle;
// };

struct OptixRay
{ 
    float3 origin;
    float3 direction;
    float3 inv_direction;
    int3 sign;
};

struct OptixPathSegment
{
    OptixRay ray;
    float3 color;
    float3 throughput;
    float3 sample_dir;
    int pixelIndex;
    bool kill;
};

struct Params
{
    OptixTraversableHandle handle;
    OptixPathSegment* path_segments;
};

struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};
