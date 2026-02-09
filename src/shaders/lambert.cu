#include "shaders/lambert.h"

namespace Lambert {

    __device__ glm::vec3 BRDF(glm::vec3 material_color) {
        return material_color * INV_PI;
    }

    __device__ float PDF(glm::vec3 sample_dir, glm::vec3 surface_normal) {
        float cosTheta = max(0.0f, glm::dot(sample_dir, surface_normal));
        return max(1e-6f, cosTheta * INV_PI);
    }
    
    __device__ glm::vec3 shadePathLambert(
        int idx,
        int iter,
        int num_paths,
        int depth,
        PathSegment &path,
        const Material &material,
        glm::vec3 materialColor,
        glm::vec3 normal
    )
    {
        glm::vec3 brdf = BRDF(materialColor);
        float absdot = max(0.0f, glm::dot(path.sample_dir, normal));
        return brdf * absdot;
    }   



    __device__ void sampleHemisphere(int idx, int num_paths, int iter, int depth, PathSegment &path, thrust::default_random_engine &rng, glm::vec3 normal) {
        glm::vec3 wo = -path.ray.direction;
        glm::vec3 wi;

        wi = calculateRandomDirectionInHemisphere(normal, rng);

        path.sample_dir = wi;
    }
}