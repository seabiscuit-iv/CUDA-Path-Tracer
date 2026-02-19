#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "config.h"

__device__ glm::vec3 ACESFilm(glm::vec3 x);

__device__ glm::vec3 AgX(glm::vec3 val);