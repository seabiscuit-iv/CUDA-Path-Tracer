#include "tonemapping.h"

__device__ glm::vec3 ACESFilm(glm::vec3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

__device__ glm::vec3 AgX(glm::vec3 val) {
    val = glm::max(val, glm::vec3(1e-6f));
    
    val.r = (log2f(val.r) + 10.0f) / 16.0f;
    val.g = (log2f(val.g) + 10.0f) / 16.0f;
    val.b = (log2f(val.b) + 10.0f) / 16.0f;
    val = glm::clamp(val, 0.0f, 1.0f);

    glm::vec3 x2 = val * val;
    glm::vec3 x4 = x2 * x2;
    val = 15.5f * x4 * val - 40.14f * x4 + 31.96f * x2 * val - 6.86f * x2 + 0.429f * val + 0.0322f;

    glm::vec3 result;
    result.r = val.r * 1.1968790f - val.g * 0.0980208f - val.b * 0.0990297f;
    result.g = -val.r * 0.0528968f + val.g * 1.1519031f - val.b * 0.0989611f;
    result.b = -val.r * 0.0529716f - val.g * 0.0980434f + val.b * 1.1510736f;

    return glm::clamp(result, 0.0f, 1.0f);
}