#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "stack.h"
#include "cuda_timer.h"

#include <fmt/core.h>

#include "common.cu"
#include "myoptix.h"
#include "texture.h"

#include "shaders/lambert.cu"
#include "shaders/specular.cu"
#include "shaders/cook_torrance.cu"
#include "shaders/glass.cu"

#define M_PI 3.14159

#define ACES 1

__constant__ PathTracerOptions DEV_OPTIONS;

__device__ glm::vec3 ACESFilm(glm::vec3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

__device__ glm::vec3 AgX(glm::vec3 val) {
    // 1. Logarithmic encoding
    // Map a wide dynamic range (-10 to +6 stops) into 0..1
    val = glm::max(val, 1e-6f); // Safety for log
    
    // Manual log2 and encoding
    val.r = (log2f(val.r) + 10.0f) / 16.0f;
    val.g = (log2f(val.g) + 10.0f) / 16.0f;
    val.b = (log2f(val.b) + 10.0f) / 16.0f;
    val = glm::clamp(val, 0.0f, 1.0f);

    // 2. AgX Sigmoid Curve (The "Look")
    glm::vec3 x2 = val * val;
    glm::vec3 x4 = x2 * x2;
    val = 15.5f * x4 * val - 40.14f * x4 + 31.96f * x2 * val - 6.86f * x2 + 0.429f * val + 0.0322f;

    // 3. Manual Matrix Multiply (AgX to Linear sRGB)
    // This avoids any GLM column/row major confusion
    glm::vec3 result;
    result.r = val.r * 1.1968790f - val.g * 0.0980208f - val.b * 0.0990297f;
    result.g = -val.r * 0.0528968f + val.g * 1.1519031f - val.b * 0.0989611f;
    result.b = -val.r * 0.0529716f - val.g * 0.0980434f + val.b * 1.1510736f;

    return glm::clamp(result, 0.0f, 1.0f);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, float iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        float invIter = __frcp_rn(iter);

        pix = pix * invIter;
        
        if (DEV_OPTIONS.color_mode == 0) {
            pix = pix / (pix + glm::vec3(1.0f));
        }
        else if (DEV_OPTIONS.color_mode == 1) {
            pix = AgX(pix);
        }
        else {
            pix = ACESFilm(pix);
        }

        //gamma correction
        pix = glm::pow(pix, glm::vec3(0.45f));

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths_A = NULL;
static PathSegment* dev_paths_B = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static int* dev_material_ids; //for optix

static glm::vec3** dev_vertex_buffer_locs;
static Triangle** dev_triangle_buffer_locs;
static glm::vec3** dev_normal_buffer_locs;
static glm::vec2** dev_uv_buffer_locs;

// Optix
static CUdeviceptr d_optix_paramters;

static uint32_t* dev_morton_codes;
static bool* dev_hit_geom;
static int* dev_path_scatter_buf;

static cudaArray_t dev_exr_array;
static cudaTextureObject_t exr_texture = 0;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths_A, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_B, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_morton_codes, pixelcount * sizeof(uint32_t));

    cudaMalloc(&dev_hit_geom, pixelcount * sizeof(bool));

    cudaMalloc(&dev_path_scatter_buf, pixelcount * sizeof(int));

    cudaMalloc( reinterpret_cast<void**>( &d_optix_paramters ), sizeof( Params ) );

    cudaMalloc( &dev_material_ids, sizeof(int) * scene->geoms.size());
    std::vector<int> material_ids;
    for (Geom& geom : scene->geoms) {
        material_ids.push_back(geom.materialid);
    }
    cudaMemcpy(dev_material_ids, material_ids.data(), material_ids.size() * sizeof(int), cudaMemcpyHostToDevice);


    std::vector<glm::vec3*> vertex_buffer_locs;
    std::vector<Triangle*> triangle_buffer_locs;
    std::vector<glm::vec3*> normal_buffer_locs;
    std::vector<glm::vec2*> uv_buffer_locs;

    for (const Geom& geo : scene->geoms) {
        if (geo.type == GeomType::MESH) {
            vertex_buffer_locs.push_back(geo.mesh.d_verts);
            triangle_buffer_locs.push_back(geo.mesh.d_triangles);
            normal_buffer_locs.push_back(geo.mesh.has_normal_buffers ? geo.mesh.d_normals : nullptr);
            uv_buffer_locs.push_back(geo.mesh.has_uvs ? geo.mesh.d_uvs : nullptr);
        }
    }

    cudaMalloc( &dev_vertex_buffer_locs, sizeof(glm::vec3*) * vertex_buffer_locs.size() );
    cudaMalloc( &dev_triangle_buffer_locs, sizeof(Triangle*) * triangle_buffer_locs.size() );
    cudaMalloc( &dev_normal_buffer_locs, sizeof(glm::vec3*) * normal_buffer_locs.size() );
    cudaMalloc( &dev_uv_buffer_locs, sizeof(glm::vec2*) * uv_buffer_locs.size() );

    cudaMemcpy( dev_vertex_buffer_locs, vertex_buffer_locs.data(), sizeof(glm::vec3*) * vertex_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_triangle_buffer_locs, triangle_buffer_locs.data(), sizeof(Triangle*) * triangle_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_normal_buffer_locs, normal_buffer_locs.data(), sizeof(glm::vec3*) * normal_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_uv_buffer_locs, uv_buffer_locs.data(), sizeof(glm::vec2*) * uv_buffer_locs.size(), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(DEV_OPTIONS, PathTracerOptions::Get(), sizeof(PathTracerOptions));

    if (!scene->exr_data.empty()) {
        // exr loading on GPU
        cudaChannelFormatDesc exr_channel_desc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&dev_exr_array, &exr_channel_desc, scene->exr_width, scene->exr_height);
        cudaMemcpyToArray(dev_exr_array, 0, 0, scene->exr_data.data(), scene->exr_width * scene->exr_height * sizeof(glm::vec4), cudaMemcpyHostToDevice);

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = dev_exr_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        cudaCreateTextureObject(&exr_texture, &res_desc, &tex_desc, nullptr);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths_A);
    cudaFree(dev_paths_B);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_morton_codes);
    cudaFree(dev_hit_geom);
    cudaFree(dev_path_scatter_buf);
    cudaFree(dev_material_ids);

    cudaFree(dev_vertex_buffer_locs);
    cudaFree(dev_triangle_buffer_locs);
    cudaFree(dev_normal_buffer_locs);
    cudaFree(dev_uv_buffer_locs);
    
    cudaFree(reinterpret_cast<void*>(d_optix_paramters));

    // free exr
    cudaDestroyTextureObject(exr_texture);
    cudaFreeArray(dev_exr_array);

    checkCUDAError("pathtraceFree");
}


__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* __restrict__ pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f);
        segment.throughput = glm::vec3(1.0f);
        segment.kill = false;

        
        CREATE_RANDOM_ENGINE(iter, index, traceDepth, u01, rng);

        float x1 = u01(rng) - 0.5f;
        float x2 = u01(rng) - 0.5f;

        float pX = (float(x) + x1 + 0.5f) - (float)cam.resolution.x * 0.5f;
        float pY = (float(y) + x2 + 0.5f) - (float)cam.resolution.y * 0.5f;

        segment.ray.direction = glm::normalize(
            cam.view 
            - (cam.right * cam.pixelLength.x * pX) 
            - (cam.up    * cam.pixelLength.y * pY)
        );

        segment.pixelIndex = index;
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    const PathSegment* __restrict__ pathSegments,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        const PathSegment pathSegment = pathSegments[path_index];
        ShadeableIntersection isect = intersections[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom &geom = geoms[i];

            if (geom.type == CUBE)
            {
                // 94 vgprs
                #if ENABLE_BOX_INTERSECTION
                    t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                #else 
                    t = -1.0f;
                #endif
            }
            else if (geom.type == SPHERE)
            {
                // 78 vgprs
                #if ENABLE_SPHERE_INTERSECTION
                    t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                #else 
                    t = -1.0f;
                #endif
            }
            else if (geom.type == MESH)
            {
                // 80 VGPRs
                #if ENABLE_MESH_INTERSECTION
                    t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                #else
                    t = -1.0f;
                #endif
            }

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            isect.t = -1.0f;
        }
        else
        {
            // The ray hits something
            isect.t = t_min;
            isect.materialId = geoms[hit_geom_index].materialid;
            isect.surfaceNormal = normal;
        }

        intersections[path_index] = isect;
    }
}

__global__ void shadePath(
    int iter,
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    Material* __restrict__ materials,
    ShadeableIntersection* __restrict__ shadeableIntersections,
    int depth,
    bool has_exr,
    cudaTextureObject_t exr,
    TextureData* textures
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    ShadeableIntersection &intersection = shadeableIntersections[idx];
    PathSegment &path = pathSegments[idx];

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

    if (intersection.t > 0.0f)
    {
        Material &material = materials[intersection.materialId];

        glm::vec3 materialColor = material.color;

        if(material.albedo_tex >= 0) {
            glm::vec2 uv = intersection.uvs;

            uv *= material.albedo_tex_transform.scale;
            
            if(material.albedo_tex_transform.rotation) {
                float c = cosf(material.albedo_tex_transform.rotation);
                float s = sinf(material.albedo_tex_transform.rotation);

                uv = glm::vec2 (
                    c * uv.x - s * uv.y,
                    s * uv.x + c * uv.y
                );
            }

            uv += material.albedo_tex_transform.offset;

            float4 tex = tex2D<float4>(textures[material.albedo_tex].tex, uv.x, uv.y);
            materialColor = glm::pow(glm::vec3(tex.x, tex.y, tex.z), glm::vec3(2.2f));
        }

        glm::vec3 normal = intersection.surfaceNormal;
        glm::vec3 normal_map;
        // if (material.normal_tex >= 0 && !DEV_OPTIONS.material_debug_mode) {
        if (material.normal_tex >= 0) {
            glm::vec2 uv = intersection.uvs;

            uv *= material.normal_tex_transform.scale;
            
            if(material.normal_tex_transform.rotation) {
                float c = cosf(material.normal_tex_transform.rotation);
                float s = sinf(material.normal_tex_transform.rotation);

                uv = glm::vec2 (
                    c * uv.x - s * uv.y,
                    s * uv.x + c * uv.y
                );
            }

            uv += material.normal_tex_transform.offset;


            float4 tex = tex2D<float4>(textures[material.normal_tex].tex, uv.x, uv.y);
            glm::vec3 local_normal = glm::vec3(tex.x, tex.y, tex.z);
            normal_map = local_normal;
            local_normal.x = local_normal.r * 2.0f - 1.0f;
            local_normal.y = local_normal.g * 2.0f - 1.0f;
            local_normal.z = local_normal.b * 2.0f - 1.0f;

            glm::vec3 bitangent = glm::normalize(glm::cross(intersection.surfaceNormal, intersection.surfaceTangent));

            glm::mat3 TBN = glm::mat3(intersection.surfaceTangent, bitangent, intersection.surfaceNormal);

            normal = glm::normalize(TBN * local_normal);
        }

        if (DEV_OPTIONS.material_debug_mode) {
            Lambert::sampleHemisphere(idx, num_paths, iter, depth, path, rng, normal);

            if (material.material_type == MaterialType::Emissive) {
                path.color += path.throughput * material.emittance * materialColor;
                path.kill = true;
            }
            else {
            Lambert::shadePathLambert(idx, iter, num_paths, depth, path, material, materialColor, normal);
            }
        }
        else {
            if (material.material_type == MaterialType::Emissive || material.material_type == MaterialType::Diffuse) {
                Lambert::sampleHemisphere(idx, num_paths, iter, depth, path, rng, normal);
            } 
            else if (material.material_type == MaterialType::Specular) {
                PerfectSpecular::sampleMirror(path, normal);
            }
            else if (material.material_type == MaterialType::Microfacet) {
                CookTorrance::sampleCookTorrance(path, material, idx, iter, depth, -path.ray.direction, intersection.surfaceNormal, material.roughness, rng, materialColor);
            }
            else if (material.material_type == MaterialType::Glass) {
                TransmissiveGlass::sampleGlass(path, material, rng, normal);
            }

            if (material.material_type == MaterialType::Emissive) {
                path.color += path.throughput * material.emittance * materialColor;
                path.kill = true;
            } 
            else if (material.material_type == MaterialType::Diffuse) {
                Lambert::shadePathLambert(idx, iter, num_paths, depth, path, material, materialColor, normal);
            } 
            else if (material.material_type == MaterialType::Specular) {
                PerfectSpecular::shadePathSpecular(path, material, materialColor);
            }
            else if (material.material_type == MaterialType::Microfacet) {
                CookTorrance::shadePathCookTorrance(path, material, materialColor, normal);
            }
            else if (material.material_type == MaterialType::Glass) {
                TransmissiveGlass::shadePathGlass(path, material, materialColor);
            }
        }
    }   
    else if (!path.kill && has_exr) {
        // hdri
        glm::vec3 d = glm::normalize(path.ray.direction);
        float phi   = atan2f(d.z, d.x);       // [-pi, pi]
        float theta = glm::acos(glm::clamp(d.y, -1.0f, 1.0f)); // [0, pi]

        float u = (phi + M_PI) * (1.0f / (2.0f * M_PI));
        float v = theta * (1.0f / M_PI);

        float4 env = tex2D<float4>(exr, u, v);

        path.color += path.throughput * glm::vec3(env.x, env.y, env.z) * DEV_OPTIONS.envmap_intensity;
    }

    if (intersection.t == -1.0f) {
        path.kill = true;
    }
    else {
        Ray& ray = path.ray;
        glm::vec3 hit_point = getPointOnRay(ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;
        ray.direction = path.sample_dir;

        float eps = 1e-4f;
        if (glm::dot(ray.direction, normal) > 0.0f) {
            ray.origin = hit_point + (normal * eps);
        }
        else {
            ray.origin = hit_point - (normal * eps);
        }   
    }
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* __restrict__ iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        glm::vec3 color = iterationPath.color;

        float maxIntensity = 1000.0f;
        float luminance = glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
        if (luminance > maxIntensity) {
            color *= (maxIntensity / luminance);
        }

        image[iterationPath.pixelIndex] += color;
    }
}


// for stream compaction
struct path_terminated {
    __host__ __device__ bool operator()(PathSegment &path) const {
        return !path.kill;
    }
};

struct sort_materials {
    __host__ __device__ bool operator()(const ShadeableIntersection &sA, const ShadeableIntersection &sB) const {
        return sA.materialId < sB.materialId;
    }
};


struct sort_rays {
    const Geom* dev_mesh;

    sort_rays(const Geom* dev_geom)
        : dev_mesh(dev_geom)
    {}

    __device__ bool operator()(const PathSegment &path) const {
        Ray r = path.ray;

        r.origin = glm::vec3(dev_mesh->inverseTransform * glm::vec4(r.origin, 1.0f));
        r.direction = glm::vec3(dev_mesh->inverseTransform * glm::vec4(r.direction, 0.0f));

        r.inv_direction.x = __frcp_rn(r.direction.x);
        r.inv_direction.y = __frcp_rn(r.direction.y);
        r.inv_direction.z = __frcp_rn(r.direction.z);

        r.sign.x = (r.inv_direction.x < 0.0f) ? 1 : 0;
        r.sign.y = (r.inv_direction.y < 0.0f) ? 1 : 0;
        r.sign.z = (r.inv_direction.z < 0.0f) ? 1 : 0;

        float t;
        return dev_mesh->mesh.bvh.dev_bvh[0].box.RayBoxInterection(r, t);
    }
};


#define MORTON_INTERP_DIST 1.0f

__device__ inline uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ inline uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);

    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);

    return (xx << 2) | (yy << 1) | zz;
}

__device__ void normalizePoint(const glm::vec3& point, glm::vec3& out, const float scene_extent) {
    out.x = (point.x + scene_extent) / (2.0f * scene_extent);
    out.y = (point.y + scene_extent) / (2.0f * scene_extent);
    out.z = (point.z + scene_extent) / (2.0f * scene_extent);
}

__device__ void normalizeDirection(const glm::vec3& dir, glm::vec3& out) {
    out = (dir + glm::vec3(1.0f)) * 0.5f;
}

__device__ uint32_t rayMortonCode(const Ray& ray, const float scene_extent) {
    glm::vec3 p0_n, p1_n;
    
    normalizePoint(ray.origin, p0_n, scene_extent);
    
    glm::vec3 farPoint = ray.origin + (ray.direction * MORTON_INTERP_DIST); 
    normalizePoint(farPoint, p1_n, scene_extent);

    glm::vec3 midpoint = 0.5f * (p0_n + p1_n);

    return morton3D(midpoint.x, midpoint.y, midpoint.z);
}

__global__ void intersectionPrecompute(int n, PathSegment* __restrict__ pathSegments, const Geom* mesh, uint32_t* morton_codes, bool* hit_geoms) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < n)
    {
        Ray r = pathSegments[path_index].ray;

        r.origin = glm::vec3(mesh->inverseTransform * glm::vec4(r.origin, 1.0f));
        r.direction = glm::vec3(mesh->inverseTransform * glm::vec4(r.direction, 0.0f));

        r.inv_direction.x = __frcp_rn(r.direction.x);
        r.inv_direction.y = __frcp_rn(r.direction.y);
        r.inv_direction.z = __frcp_rn(r.direction.z);

        r.sign.x = (r.inv_direction.x < 0.0f) ? 1 : 0;
        r.sign.y = (r.inv_direction.y < 0.0f) ? 1 : 0;
        r.sign.z = (r.inv_direction.z < 0.0f) ? 1 : 0;

        BoundingBox bbox = mesh->mesh.bvh.dev_bvh[0].box;

        float scene_extent = glm::length(bbox.box_max - bbox.box_min);

        float t;
        morton_codes[path_index] = rayMortonCode(r, scene_extent);
        hit_geoms[path_index] = bbox.RayBoxInterection(r, t);
    }
} 




struct sort_rays_morton {
    const uint32_t* d_morton_codes;
    const bool* d_hit_geoms;

    sort_rays_morton(const uint32_t* d_m_c, const bool* d_h_g) :
        d_morton_codes(d_m_c),
        d_hit_geoms(d_h_g)
    {}

    __device__ bool operator()(int pA, int pB) const {
        bool b = !d_hit_geoms[pB];
        if (b || !d_hit_geoms[pA]) {
            return b;
        }

        return d_morton_codes[pA] < d_morton_codes[pB];
    }
};



__global__ void drawBVH(
    int depth,
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment &pathSegment = pathSegments[path_index];

        int count = 0;

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom &geom = geoms[i];

            if (geom.type == MESH)
            {
                count += bvhCountHits(geom, pathSegment.ray);
            }
        }

        pathSegment.color += float(count) * glm::vec3(0.001f);
    }
}



void pathtrace(uchar4* pbo, int frame, int iter)
{
    // fmt::println("PATHTRACE: {} vs {}", sizeof(ShadeableIntersection), sizeof(OptixShadeableIntersection));
    // fmt::println("Offset 0: {} vs {}", offsetof(ShadeableIntersection, t), offsetof(OptixShadeableIntersection, t));
    // fmt::println("Offset 1: {} vs {}", offsetof(ShadeableIntersection, surfaceNormal), offsetof(OptixShadeableIntersection, surfaceNormal));
    // fmt::println("Offset 2: {} vs {}", offsetof(ShadeableIntersection, surfaceTangent), offsetof(OptixShadeableIntersection, surfaceTangent));
    // fmt::println("Offset 3: {} vs {}", offsetof(ShadeableIntersection, materialId), offsetof(OptixShadeableIntersection, materialId));
    // fmt::println("Offset 4: {} vs {}", offsetof(ShadeableIntersection, uvs), offsetof(OptixShadeableIntersection, u));

    const int traceDepth = PathTracerOptions::Get()->material_debug_mode ? 2 : hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = BLOCK_SIZE_1D;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths_A);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    CudaTimer cudaTimer;

    PathSegment* dev_paths;
    PathSegment* dev_paths_sorted;

    int last_num_paths = num_paths;

    bool iterationComplete = false;
    while (!iterationComplete)
    { 
        dev_paths = (depth % 2) == 0 ? dev_paths_A : dev_paths_B;
        dev_paths_sorted = (depth % 2) == 1 ? dev_paths_A : dev_paths_B;

        cudaTimer.record(fmt::format("Start, Iter {}", depth+1));

        if (iter == MAX_ITERATIONS) {
            exit(0);
        }

        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
        thrust::sequence(dPtr(dev_path_scatter_buf), dPtr(dev_path_scatter_buf) + num_paths);

        cudaTimer.record(fmt::format("Memset, Iter {}", depth+1));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        #if 0
            const Geom* d_mesh;
            for (int i = 0; i < hst_scene->geoms.size(); i++) {
                Geom &geom = hst_scene->geoms[i];
                if (geom.type == GeomType::MESH) {
                    d_mesh = dev_geoms + i;
                    break;
                }
            }

            if (d_mesh == nullptr) {
                printf("ERROR: No Mesh Detected\n");
                exit(1);
            }

            thrust::partition(dPtr(dev_paths), dPtr(dev_paths) + num_paths, sort_rays(d_mesh));

            cudaTimer.record(fmt::format("Partition Mesh Hits, Iter {}", depth+1)); 
        #elif 1
            const Geom* d_mesh;
            for (int i = 0; i < hst_scene->geoms.size(); i++) {
                Geom &geom = hst_scene->geoms[i];
                if (geom.type == GeomType::MESH) {
                    d_mesh = dev_geoms + i;
                    break;
                }
            }

            if (d_mesh == nullptr) {
                printf("ERROR: No Mesh Detected\n");
                exit(1);
            }

            intersectionPrecompute<<<numblocksPathSegmentTracing, blockSize1d>>> (
                num_paths,
                dev_paths,
                d_mesh,
                dev_morton_codes,
                dev_hit_geom
            );

            cudaTimer.record(fmt::format("Morton Precompute, Iter {}", depth+1));

            thrust::sort(dPtr(dev_path_scatter_buf), dPtr(dev_path_scatter_buf) + num_paths, sort_rays_morton(dev_morton_codes, dev_hit_geom));
            thrust::gather(dPtr(dev_path_scatter_buf), dPtr(dev_path_scatter_buf) + num_paths, dPtr(dev_paths), dPtr(dev_paths_sorted));
            thrust::copy(dPtr(dev_paths) + num_paths, dPtr(dev_paths) + last_num_paths, dPtr(dev_paths_sorted) + num_paths);

            cudaTimer.record(fmt::format("Sort Mesh Hits Morton, Iter {}", depth+1));
        #endif


        if (PathTracerOptions::Get()->debug_bvh) {
            drawBVH<<<numblocksPathSegmentTracing, blockSize1d>>> (
                depth,
                num_paths,
                dev_paths_sorted,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
            );
        }
        else {
            #if !OPTIX
                // tracing
                computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
                    depth,
                    num_paths,
                    dev_paths_sorted,
                    dev_geoms,
                    hst_scene->geoms.size(),
                    dev_intersections
                );
                checkCUDAError("compute intersections");
            #else // OPTIX
                // time for some optix magic
                Params optix_params = {};
                optix_params.handle = hst_scene->ias_handle;
                optix_params.path_segments = reinterpret_cast<OptixPathSegment*>(dev_paths_sorted);
                optix_params.debug_image = reinterpret_cast<float3*>(dev_image);
                optix_params.shadeable_intersections = reinterpret_cast<OptixShadeableIntersection*>(dev_intersections);
                optix_params.material_ids = dev_material_ids;
                optix_params.vertex_buffer_locations = (float3**)dev_vertex_buffer_locs;
                optix_params.triangle_buffer_locations = (OptixTriangle**)dev_triangle_buffer_locs;
                optix_params.normal_buffer_locations = (float3**)dev_normal_buffer_locs;
                optix_params.uv_buffer_locations = (float2**)dev_uv_buffer_locs;
                cudaMemcpy(reinterpret_cast<void*>(d_optix_paramters), &optix_params, sizeof(Params), cudaMemcpyHostToDevice);
                OPTIX_CHECK(
                    optixLaunch(hst_scene->optix_pipeline, 0, d_optix_paramters, sizeof(Params), &hst_scene->optix_sbt, num_paths, 1, 1);
                );
                // fmt::println("OptixTrace Iteration {}", iter);
                // end of optix magic
                cudaTimer.record(fmt::format("Optix Compute Intersections"));
            #endif //OPTIX

            depth++;

            cudaTimer.record(fmt::format("Compute Intersections, Iter {}", depth));

            #if MATERIAL_SORTING
                thrust::sort_by_key(
                    thrust::device,
                    dev_intersections,
                    dev_intersections + num_paths,
                    dev_paths_sorted,
                    sort_materials()
                );

                cudaTimer.record(fmt::format("Material Sorting, Iter %i", depth));
            #endif

            shadePath<<<numblocksPathSegmentTracing, blockSize1d>>>(
                iter,
                num_paths,
                dev_paths_sorted,
                dev_materials,
                dev_intersections,
                depth,
                !hst_scene->exr_data.empty(),
                exr_texture,
                TextureHandler::get().dev_textures
            );

            cudaTimer.record(fmt::format("Shade Path, Iter {}", depth));
        }

        if (depth == traceDepth) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }

        #if STREAM_COMPACTION
            if (depth == 1 && !PathTracerOptions::Get()->debug_bvh) {
                auto new_end = thrust::partition(dPtr(dev_paths_sorted), dPtr(dev_paths_sorted) + num_paths, path_terminated());
                last_num_paths = num_paths;
                num_paths = new_end - dPtr(dev_paths_sorted);
                checkCUDAError("thrust::partition");

                cudaTimer.record(fmt::format("Stream Compaction, Iter {}", depth));
            }
        #endif

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        if (num_paths == 0) {
            iterationComplete = true;
        }

        cudaTimer.record(fmt::format("End, Iter {}", depth));

        if (PathTracerOptions::Get()->debug_bvh) {
            break;
        }
    }
    
    // cudaTimer.report();

    // printf("Total Iteration Elapsed Time: %f\n\n", cudaTimer.get_elapsed("Start, Iter 1", "End, Iter 8"));

    cudaTimer.clean();

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths_sorted);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, float(iter), dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
