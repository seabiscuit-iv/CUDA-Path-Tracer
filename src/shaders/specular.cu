#include "shaders/specular.h"

namespace PerfectSpecular {
    __device__ void shadePathSpecular(
        PathSegment &path,
        Material &material,
        glm::vec3 color
    )
    {
        path.throughput *= color;
    }   



    __device__ void sampleMirror(PathSegment &path, glm::vec3 normal) {
        glm::vec3 wo = -path.ray.direction;
        glm::vec3 wi = glm::reflect(-wo, normal);

        path.sample_dir = wi;
    }
}