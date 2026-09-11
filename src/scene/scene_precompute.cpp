#include "scene.h"

#include "utilities.h"

#include <cuda.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "texture.h"

#include <fstream>
#include <iostream>
#include <string>
#include <fmt/format.h>
#include <unordered_map>

using namespace std;

void Scene::precompute_emissive_mesh_area() {
    float emissive_area = 0.0f;
    int i = 0;

    emissive_geoms.clear();
    emissive_geom_area_prefix.clear();

    for (auto& geom : geoms) {
        float cumulative_geom_area = 0.0f;

        geom.mesh.h_triangle_area_percentage_prefix.clear();
        
        for(Triangle& tri : geom.mesh.h_triangles) {
            glm::vec3 A = geom.mesh.h_verts[tri.v_indices[0]];
            glm::vec3 B = geom.mesh.h_verts[tri.v_indices[1]];
            glm::vec3 C = geom.mesh.h_verts[tri.v_indices[2]];

            A = glm::vec3(geom.transform * glm::vec4(A, 1.0f));
            B = glm::vec3(geom.transform * glm::vec4(B, 1.0f));
            C = glm::vec3(geom.transform * glm::vec4(C, 1.0f));

            glm::vec3 u = B - A;
            glm::vec3 v = C - A;

            cumulative_geom_area += 0.5f * glm::length(glm::cross(u, v));
            geom.mesh.h_triangle_area_percentage_prefix.push_back(cumulative_geom_area);
        }

        for (int k = 0; k < geom.mesh.h_triangle_area_percentage_prefix.size(); k++) {
            geom.mesh.h_triangle_area_percentage_prefix[k] /= cumulative_geom_area;
        }

        #if UBER_SHADER
            if (glm::length(materials[geom.materialid].emission.emission_color) * materials[geom.materialid].emission.emission_strength > EPSILON) {
        #else
            if (materials[geom.materialid].material_type == MaterialType::Emissive) {
        #endif
            emissive_area += cumulative_geom_area;
            emissive_geoms.push_back(i);
            emissive_geom_area_prefix.push_back(emissive_area);
        }

        geom.mesh.has_triangle_area_percentage_prefix = true;
        i++;
    }

    for (int i = 0; i < emissive_geom_area_prefix.size(); i++) {
        emissive_geom_area_prefix[i] /= emissive_area;
    }

    total_emissive_mesh_area = emissive_area;
}


void Scene::precompute_hdri_emission() {
    fmt::println("Beginning Environment Map Emission Precompute");

    hdri_conditional_cdfs.resize(exr_height * exr_width);
    hdri_marginal_cdf.resize(exr_height);
    total_hdri_emission = 0.0f;

    for(int y = 0; y < exr_height; y++) {
        float row_total = 0.0f;
        float theta = PI * (y + 0.5f) / exr_height;
        
        for(int x = 0; x < exr_width; x++) {
            float emission = glm::length(exr_data[y * exr_width + x]);
            float weight = emission * sin(theta);

            row_total += weight;
            hdri_conditional_cdfs[y * exr_width + x] = row_total;
        }

        if (row_total > 0.0f) {
            for(int x = 0; x < exr_width; x++) {
                hdri_conditional_cdfs[y * exr_width + x] /= row_total;
            }
        }
        else {
            for(int x = 0; x < exr_width; x++) {
                hdri_conditional_cdfs[y * exr_width + x] = (float)(x + 1) / (float)exr_width;
            }
        }

        hdri_conditional_cdfs[y * exr_width + (exr_width - 1)] = 1.0f;

        total_hdri_emission += row_total;
        hdri_marginal_cdf[y] = total_hdri_emission;
    }

    if (total_hdri_emission == 0.0f) {
        fmt::println("HDRI has emission of 0.0f");
        exit(1);
    }

    for (int y = 0; y < exr_height; y++) {
        hdri_marginal_cdf[y] /= total_hdri_emission;
    }

    hdri_marginal_cdf[exr_height - 1] = 1.0f;

    fmt::println("End Environment Map Emission Precompute");


    // fmt::println("Beginning Environment Map Verification");

    // for(int c = 0; c < exr_height; c++) {
    //     float sum = 0.0f;
    //     for (int y = exr_width - 1; y < exr_width; y++) {
    //         sum += hdri_conditional_cdfs[y + c * exr_width];
    //     }

    //     fmt::println("EXR COL SUM: {}", sum);
    // }

    // float sum = 0.0f;
    // for (int x = exr_height - 1; x < exr_height; x++) {
    //     sum += hdri_marginal_cdf[x];
    // }
    // fmt::println("EXR ROW SUM: {}", sum);

    // fmt::println("End Environment Map Verification");
}
