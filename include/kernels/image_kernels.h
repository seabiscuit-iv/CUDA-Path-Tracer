#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "config.h"

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, float iter, glm::vec3* image);

__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* __restrict__ iterationPaths);
