#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "stack.h"

#include "common.cu"

#include "shaders/lambert.cu"
#include "shaders/specular.cu"
#include "shaders/cook_torrance.cu"


// CONFIGURATION
#define STREAM_COMPACTION 1
#define MATERIAL_SORTING 0  // enable this if you have a high number of materials

// Set this to -1 when profiling off
#define MAX_ITERATIONS -1

// Bump the shader version to recompile shaders. We need a better solution for this
#define SHADER_VER 2.8

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
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

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
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                // 78 vgprs
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                // 80 VGPRs
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        if (iter == MAX_ITERATIONS) {
            exit(0);
        }

        // if (iter == 1) {
            // printf("NumPaths: %d\n", num_paths);
        // }

        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("compute intersections");
        depth++;

        #if MATERIAL_SORTING
            thrust::sort_by_key(
                thrust::device,
                dev_intersections,
                dev_intersections + num_paths,
                dev_paths,
                sort_materials()
            );
        #endif

        shadePath<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_paths,
            dev_materials,
            dev_intersections,
            depth
        );

        if (depth == traceDepth) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }

        #if STREAM_COMPACTION
            auto new_end = thrust::partition(dPtr(dev_paths), dPtr(dev_paths) + num_paths, path_terminated());
            num_paths = new_end - dPtr(dev_paths);
            checkCUDAError("thrust::remove_if");
        #endif // STREAM_COMPACTION

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        if (num_paths == 0) {
            iterationComplete = true;
        }
    }

        // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, float(iter), dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
