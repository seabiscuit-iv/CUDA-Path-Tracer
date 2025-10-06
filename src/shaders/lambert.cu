#ifndef LAMBERT
#define LAMBERT

#include "common.cu"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#define INV_PI 0.3183098f

namespace Lambert {

    __device__ glm::vec3 BRDF(glm::vec3 material_color) {
        return material_color * INV_PI;
    }

    __device__ float PDF(glm::vec3 sample_dir, glm::vec3 surface_normal) {
        float cosTheta = max(0.0f, glm::dot(sample_dir, surface_normal));
        return cosTheta * INV_PI;
    }
    
    __device__ void shadePathLambert(
        int idx,
        int iter,
        int num_paths,
        int depth,
        ShadeableIntersection &intersection,
        PathSegment &path,
        const Material &material
    )
    {
        glm::vec3 materialColor = material.color;

        glm::vec3 brdf = BRDF(materialColor);
        float absdot = max(0.0f, glm::dot(path.sample_dir, intersection.surfaceNormal));
        float pdf = max(1e-6f, PDF(path.sample_dir, intersection.surfaceNormal));
        path.throughput *= brdf * absdot / pdf;
    }   



__device__ void sampleHemisphere(int idx, int num_paths, int iter, int depth, PathSegment &path, ShadeableIntersection &intersection) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

        glm::vec3 wo = -path.ray.direction;
        glm::vec3 wi;

        wi = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);

        path.sample_dir = wi;
    }
}

#endif // LAMBERT