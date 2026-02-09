#ifndef LAMBERT
#define LAMBERT

#include "common.h"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#define INV_PI 0.3183098f

namespace Lambert {

    __device__ glm::vec3 BRDF(glm::vec3 material_color);

    __device__ float PDF(glm::vec3 sample_dir, glm::vec3 surface_normal);
    
    __device__ glm::vec3 shadePathLambert(
        int idx,
        int iter,
        int num_paths,
        int depth,
        PathSegment &path,
        const Material &material,
        glm::vec3 materialColor,
        glm::vec3 normal
    );

    __device__ void sampleHemisphere(int idx, int num_paths, int iter, int depth, PathSegment &path, thrust::default_random_engine &rng, glm::vec3 normal);
}

#endif // LAMBERT