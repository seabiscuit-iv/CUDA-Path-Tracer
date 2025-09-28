#ifndef LAMBERT
#define LAMBERT

#include "common.cu"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace Lambert {

    __device__ glm::vec3 BRDF_Lambert(glm::vec3 material_color) {
        return material_color / glm::pi<float>();
    }

    __device__ float PDF_Lambert(glm::vec3 sample_dir, glm::vec3 surface_normal) {
        float cosTheta = max(0.0f, glm::dot(sample_dir, surface_normal));
        return cosTheta / glm::pi<float>();
    }
    
    __global__ void shadePathLambert(
        int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials,
        int depth
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_paths)
        {
            ShadeableIntersection &intersection = shadeableIntersections[idx];
            PathSegment &path = pathSegments[idx];
            if (intersection.t > 0.0f) // if the intersection exists...
            {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
                thrust::uniform_real_distribution<float> u01(0, 1);

                Material material = materials[intersection.materialId];
                glm::vec3 materialColor = material.color;

                // If the material indicates that the object was a light, "light" the ray
                if (material.emittance > 0.0f) {
                    path.color += path.throughput * material.emittance * material.color;
                }
                else {
                    glm::vec3 brdf = BRDF_Lambert(materialColor);
                    float absdot = max(0.0f, glm::dot(path.sample_dir, intersection.surfaceNormal));
                    float pdf = max(1e-6f, PDF_Lambert(path.sample_dir, intersection.surfaceNormal));
                    path.throughput *= brdf * absdot / pdf;
                }
            }
        }
    }   



    __global__ void sampleHemisphere(int num_paths, int iter, int depth, PathSegment* paths, ShadeableIntersection *intersections) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_paths)
        {
            return;
        }

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

        glm::vec3 wo = -paths[idx].ray.direction;
        glm::vec3 wi;

        wi = calculateRandomDirectionInHemisphere(intersections[idx].surfaceNormal, rng);

        paths[idx].sample_dir = wi;
    }
}

#endif // LAMBERT