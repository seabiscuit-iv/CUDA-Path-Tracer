#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

