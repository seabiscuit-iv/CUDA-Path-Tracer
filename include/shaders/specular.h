#ifndef SPECULAR
#define SPECULAR

#include "common.h"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace PerfectSpecular {
    __device__ void shadePathSpecular(
        PathSegment &path,
        const Material &material,
        glm::vec3 color
    );

    __device__ void sampleMirror(PathSegment &path, glm::vec3 normal);
}

#endif // SPECULAR