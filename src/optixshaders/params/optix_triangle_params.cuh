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

struct OptixShadeableIntersection
{
  float t;                  // 0
  float3 surfaceNormal;     // 4 (CUDA float3 at offset 4 is fine here)
  int materialId;           // 16
  float _pad0;              // 20
  float u;               // 24
  float v;
};

struct OptixTriangle {
    unsigned int v_indices[3];
    unsigned int n_indices[3];
    unsigned int uv_indices[3];
};


struct Params
{
    OptixTraversableHandle handle;
    OptixPathSegment* path_segments;
    float3* debug_image;
    OptixShadeableIntersection* shadeable_intersections;    
    int* material_ids;

    float3** vertex_buffer_locations;
    OptixTriangle** triangle_buffer_locations;
    float3** normal_buffer_locations;
    float2** uv_buffer_locations;
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
