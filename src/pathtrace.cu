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

#include "shaders/lambert.cu"
#include "shaders/specular.cu"
#include "shaders/cook_torrance.cu"


// CONFIGURATION
#define STREAM_COMPACTION 0
#define MATERIAL_SORTING 0  // enable this if you have a high number of materials

// Set this to -1 when profiling off
#define MAX_ITERATIONS -1

// Bump the shader version to recompile shaders. We need a better solution for this
#define SHADER_VER 2.8

#define DRAW_BVH 0

#define OPTIX 1

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
        
        // reinhard op
        // pix = pix / (pix + glm::vec3(1.0f));

        //gamma correction
        // pix = glm::pow(pix, glm::vec3(0.45f));

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

// Optix
static CUdeviceptr d_optix_paramters;

static uint32_t* dev_morton_codes;
static bool* dev_hit_geom;
static int* dev_path_scatter_buf;

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

    for (const Geom& geo : scene->geoms) {
        if (geo.type == GeomType::MESH) {
            vertex_buffer_locs.push_back(geo.mesh.d_verts);
            triangle_buffer_locs.push_back(geo.mesh.d_triangles);
            normal_buffer_locs.push_back(geo.mesh.has_normal_buffers ? geo.mesh.d_normals : nullptr);
        }
    }

    cudaMalloc( &dev_vertex_buffer_locs, sizeof(glm::vec3*) * vertex_buffer_locs.size() );
    cudaMalloc( &dev_triangle_buffer_locs, sizeof(Triangle*) * triangle_buffer_locs.size() );
    cudaMalloc( &dev_normal_buffer_locs, sizeof(glm::vec3*) * normal_buffer_locs.size() );

    cudaMemcpy( dev_vertex_buffer_locs, vertex_buffer_locs.data(), sizeof(glm::vec3*) * vertex_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_triangle_buffer_locs, triangle_buffer_locs.data(), sizeof(Triangle*) * triangle_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_normal_buffer_locs, normal_buffer_locs.data(), sizeof(glm::vec3*) * normal_buffer_locs.size(), cudaMemcpyHostToDevice);

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
    
    cudaFree(reinterpret_cast<void*>(d_optix_paramters));

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
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.kill = false;

        
        CREATE_RANDOM_ENGINE(iter, index, traceDepth, u01, rng);

        float x1 = u01(rng) - 0.5f;
        float x2 = u01(rng) - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float(x) + x1) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float(y) + x2) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
    }
}


#define ENABLE_BOX_INTERSECTION     1
#define ENABLE_SPHERE_INTERSECTION  0
#define ENABLE_MESH_INTERSECTION    1


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
    int depth
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
        if (material.material_type == MaterialType::Emissive || material.material_type == MaterialType::Diffuse) {
            Lambert::sampleHemisphere(idx, num_paths, iter, depth, path, intersection, rng);
        } 
        else if (material.material_type == MaterialType::Specular) {
            PerfectSpecular::sampleMirror(path, intersection);
        }
        else if (material.material_type == MaterialType::Microfacet) {
            CookTorrance::sampleCookTorrance(path, material, idx, iter, depth, -path.ray.direction, intersection.surfaceNormal, material.roughness, rng);
        }

        if (material.material_type == MaterialType::Emissive) {
            path.color += path.throughput * material.emittance * material.color;
            path.kill = true;
        } 
        else if (material.material_type == MaterialType::Diffuse) {
            Lambert::shadePathLambert(idx, iter, num_paths, depth, intersection, path, material);
        } 
        else if (material.material_type == MaterialType::Specular) {
            PerfectSpecular::shadePathSpecular(path, material);
        }
        else if (material.material_type == MaterialType::Microfacet) {
            CookTorrance::shadePathCookTorrance(intersection, path, material);
        }
    }

    if (intersection.t == -1.0f) {
        path.kill = true;
    }
    else {
        Ray &ray = path.ray;
        ray.origin = getPointOnRay(ray, intersection.t);
        ray.direction = path.sample_dir;
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
        image[iterationPath.pixelIndex] += iterationPath.color;
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
    const int traceDepth = hst_scene->state.traceDepth;
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


        #if DRAW_BVH
            drawBVH<<<numblocksPathSegmentTracing, blockSize1d>>> (
                depth,
                num_paths,
                dev_paths_sorted,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
            );
        #else
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
                depth
            );

            cudaTimer.record(fmt::format("Shade Path, Iter {}", depth));
        #endif

        if (depth == traceDepth) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }

        #if STREAM_COMPACTION && !DRAW_BVH
            if (depth == 1) {
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

        #if DRAW_BVH
            break;
        #endif
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
