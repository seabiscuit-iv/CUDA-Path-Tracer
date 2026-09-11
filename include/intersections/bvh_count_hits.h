#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

__device__ int bvhCountHits(
    const Geom &mesh,
    Ray r
);
