#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "config.h"
#include "sceneStructs.h"
#include "texture.h"

__device__ glm::vec3 get_albedo(const Material& material, glm::vec2 uv, const TextureData* textures);


__device__ glm::vec3 get_normal(const Material& material, const TextureData* textures, const ShadeableIntersection& intersection, glm::vec3* out_normal_map);