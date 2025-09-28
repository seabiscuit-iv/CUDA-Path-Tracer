#ifndef SPECULAR
#define SPECULAR

#include "common.cu"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace PerfectSpecular {
    __device__ void shadePathSpecular(
        PathSegment &path,
        Material &material
    )
    {
        path.throughput *= material.color;
    }   



    __device__ void sampleMirror(PathSegment &path, ShadeableIntersection &intersection) {
        glm::vec3 wo = -path.ray.direction;
        glm::vec3 wi = glm::reflect(-wo, intersection.surfaceNormal);

        path.sample_dir = wi;
    }
}

#endif // SPECULAR