#pragma once

#include "mesh/bounding_box.h"
#include "mesh/bvh.h"

#include <glm/glm.hpp>
#include <vector>

#include "sceneStructs.h"

// START AND END ARE INCLUSIVE
BoundingBox fill_bvh(int idx, int start, int end, std::vector<BVHNode> &h_bvh, const std::vector<glm::vec3> &verts, const std::vector<Triangle> &triangles, std::vector<int> &tri_indices, int &allocated_length);
