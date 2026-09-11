#include "kernels/shade_path.h"

#include "pathtrace/dev_options.h"
#include "common.h"
#include "utilities.h"
#include "interactions.h"
#include "material_queries.h"
#include "material_debug_render.h"
#include "sample_materials.h"
#include "update_throughput_materials.h"

#include "shaders/lambert.h"
#include "shaders/specular.h"
#include "shaders/cook_torrance.h"
#include "shaders/glass.h"

#include <glm/gtx/norm.hpp>
#include <thrust/random.h>

#define M_PI 3.14159

__device__ float envmap_pdf(glm::vec3 d, float* marginal_cdf, float* conditional_cdfs, int W, int H) {
    float phi   = atan2f(d.z, d.x);
    float theta = acosf(glm::clamp(d.y, -1.0f, 1.0f));

    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;

    int col = glm::clamp(int(u * W), 0, W - 1);
    int row = glm::clamp(int(v * H), 0, H - 1);

    float p_marginal    = (marginal_cdf[row] - (row == 0 ? 0.0f : marginal_cdf[row - 1])) * H;
    float p_conditional = (conditional_cdfs[row * W + col] - (col == 0 ? 0.0f : conditional_cdfs[row * W + col - 1])) * W;

    float sin_theta = sinf(theta);
    return (sin_theta > 1e-6f) ? (p_marginal * p_conditional) / (2.0f * PI * PI * sin_theta) : 0.0f;
}


__global__ void shadePath(
    int iter,
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    Material* __restrict__ materials,
    ShadeableIntersection* __restrict__ shadeableIntersections,
    ShadeableIntersection* __restrict__ directLightIntersections,
    ShadeableIntersection* __restrict__ environmentMapIntersections,
    int depth,
    bool has_exr,
    cudaTextureObject_t exr,
    TextureData* textures,
    int num_emissive_geoms,
    int* emissive_geoms,
    float* emissive_geoms_area_prefix,
    const Geom* __restrict__ geoms,
    float total_emissive_mesh_area,
    float* marginal_cdf,
    float* conditional_cdfs,
    int exr_width,
    int exr_height
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    ShadeableIntersection default_isect = {};
    ShadeableIntersection &intersection = shadeableIntersections[idx];
    ShadeableIntersection &direct_light_intersection = directLightIntersections[idx];
    ShadeableIntersection &env_map_intersesction = has_exr ? environmentMapIntersections[idx] : default_isect;
    PathSegment &path = pathSegments[idx];

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth);

    if (intersection.t > 0.0f)
    {
        Material &material = materials[intersection.materialId];

        glm::vec3 materialColor = get_albedo(material, intersection.uvs, textures);

        glm::vec3 normal_map;
        glm::vec3 normal = get_normal(material, textures, intersection, &normal_map);

        glm::vec2 metallic_roughness = get_metallic_roughness(material, intersection.uvs, textures);
        float roughness = metallic_roughness.x;
        float metallic = metallic_roughness.y;

        #if UBER_SHADER
            glm::vec3 emission = get_emission(material, intersection.uvs, textures);
        #endif

        #if UBER_SHADER
            bool is_specular = false;
        #else
            bool is_specular = (material.material_type == MaterialType::Specular || material.material_type == MaterialType::Glass);
        #endif

        if (DEV_OPTIONS.material_debug_mode != 0) {
            render_material_debug_mode(
                path,
                materialColor,
                normal,
                normal_map,
                roughness,
                metallic,
                DEV_OPTIONS.material_debug_mode
            );
        }
        else {
            sample_materials(
                material,
                intersection,
                path,
                idx,
                num_paths,
                iter,
                depth,
                rng,
                materialColor,
                normal,
                roughness,
                metallic
            );

            if (DEV_OPTIONS.environment_map_importance_sampling) {
                #if UBER_SHADER
                if (glm::length(emission) > EPSILON) {
                #else
                if (material.material_type == MaterialType::Emissive) {
                #endif
                    float mis_weight = 1.0f;

                    if (depth > 1 && !path.last_bounce_was_specular) {
                        float pdf_bsdf = path.last_pdf;
                        float pdf_env_of_bsdf = envmap_pdf(path.ray.direction, marginal_cdf, conditional_cdfs, exr_width, exr_height);


                        mis_weight = (pdf_bsdf * pdf_bsdf) / (pdf_bsdf * pdf_bsdf + pdf_env_of_bsdf * pdf_env_of_bsdf);
                    }

                    #if UBER_SHADER
                        path.color += path.throughput * (mis_weight * emission);
                    #else
                        path.color += path.throughput * (mis_weight * (material.emittance * materialColor));
                    #endif
                    path.kill = true;
                }
                else {
                    if (env_map_intersesction.t < 0.0f && !is_specular && !path.kill && has_exr) {
                        glm::vec3 env_dir = path.environment_map_sample;

                        float pdf_env_of_env;
                        float pdf_bsdf_of_env;

                        #if UBER_SHADER
                            pdf_bsdf_of_env = CookTorrance::PDF(-path.ray.direction, env_dir, normal, roughness, metallic, materialColor);
                        #else
                            if (material.material_type == MaterialType::Diffuse) {
                                pdf_bsdf_of_env = Lambert::PDF(env_dir, normal);
                            }
                            else if (material.material_type == MaterialType::Microfacet) {
                                pdf_bsdf_of_env = CookTorrance::PDF(-path.ray.direction, env_dir, normal, roughness, metallic, materialColor);
                            }
                        #endif

                        pdf_env_of_env = envmap_pdf(env_dir, marginal_cdf, conditional_cdfs, exr_width, exr_height);

                        float mis_weight = (pdf_env_of_env * pdf_env_of_env) / (pdf_env_of_env * pdf_env_of_env + pdf_bsdf_of_env * pdf_bsdf_of_env);

                        glm::vec3 brdf;
                        #if UBER_SHADER
                            brdf = CookTorrance::BRDF(-path.ray.direction, normal, env_dir, materialColor, roughness, metallic);
                        #else
                            if (material.material_type == MaterialType::Diffuse) {
                                brdf = materialColor / PI;
                            }
                            else if (material.material_type == MaterialType::Microfacet) {
                                brdf = CookTorrance::BRDF(-path.ray.direction, normal, env_dir, materialColor, roughness, metallic);
                            }
                        #endif

                        glm::vec3 d = glm::normalize(env_dir);
                        float phi   = atan2f(d.z, d.x);       // [-pi, pi]
                        float theta = glm::acos(glm::clamp(d.y, -1.0f, 1.0f)); // [0, pi]

                        float u = (phi + M_PI) * (1.0f / (2.0f * M_PI));
                        float v = theta * (1.0f / M_PI);

                        float4 env = tex2D<float4>(exr, u, v);
                        glm::vec3 env_light = glm::vec3(env.x, env.y, env.z) * DEV_OPTIONS.envmap_intensity;

                        float cosThetaSurface = glm::dot(normal, env_dir);

                        if (cosThetaSurface > 0.0f && pdf_env_of_env > 1e-8f) {
                            glm::vec3 contribution = (env_light * brdf * cosThetaSurface) / pdf_env_of_env;
                            path.color += path.throughput * contribution * mis_weight;
                        }
                    }
                }
            }
            else if (DEV_OPTIONS.direct_light_sampling) {
                #if UBER_SHADER
                if (glm::length(emission) > EPSILON) {
                #else
                if (material.material_type == MaterialType::Emissive) {
                #endif
                    float mis_weight = 1.0f;

                    if (depth > 1 && !path.last_bounce_was_specular) {
                        float pdf_bsdf = path.last_pdf;

                        float dist_sq = intersection.t * intersection.t;
                        float cosThetaLight = glm::dot(intersection.surfaceNormal, -path.ray.direction);

                        if (cosThetaLight > 0.0001f) {
                            float pdf_dl_area = 1.0f / total_emissive_mesh_area;
                            float pdf_dl_sa = pdf_dl_area * (dist_sq / cosThetaLight);
                            
                            mis_weight = (pdf_bsdf * pdf_bsdf) / (pdf_bsdf * pdf_bsdf + pdf_dl_sa * pdf_dl_sa);
                        }
                        else {
                            mis_weight = 0.0f;
                        }
                    }


                    #if UBER_SHADER
                        path.color += path.throughput * (mis_weight * emission);
                    #else
                        path.color += path.throughput * (mis_weight * (material.emittance * materialColor));
                    #endif
                    path.kill = true;
                } 
                else {

                    if (direct_light_intersection.t > 0.0f && !is_specular && !path.kill) {
                        glm::vec3 surface_point = intersection.t * path.ray.direction + path.ray.origin;
                        glm::vec3 light_vec = path.direct_light_sample - surface_point;

                        float dist_sq = glm::dot(light_vec, light_vec);
                        float dist = sqrt(dist_sq);
                        glm::vec3 light_dir = light_vec / dist;

                        float cosThetaSurface = glm::dot(normal, light_dir);
                        float cosThetaLight = glm::dot(direct_light_intersection.surfaceNormal, -light_dir);

                        if (cosThetaSurface > 0.0f && cosThetaLight > 0.0f) {
                            float pdf_dl_area = 1.0f / total_emissive_mesh_area;
                            float pdf_dl_sa = pdf_dl_area * (dist_sq / cosThetaLight);

                            float pdf_bsdf = 0.0f;

                            #if UBER_SHADER
                                pdf_bsdf = CookTorrance::PDF(-path.ray.direction, light_dir, normal, roughness, metallic, materialColor);
                            #else
                                if (material.material_type == MaterialType::Diffuse) {
                                    pdf_bsdf = Lambert::PDF(light_dir, normal);
                                }
                                else if (material.material_type == MaterialType::Microfacet) {
                                    pdf_bsdf = CookTorrance::PDF(-path.ray.direction, light_dir, normal, roughness, metallic, materialColor);
                                }
                            #endif

                            float mis_weight = (pdf_dl_sa * pdf_dl_sa) / (pdf_dl_sa * pdf_dl_sa + pdf_bsdf * pdf_bsdf);

                            glm::vec3 brdf;
                            #if UBER_SHADER
                                brdf = CookTorrance::BRDF(-path.ray.direction, normal, light_dir, materialColor, roughness, metallic);
                            #else
                                if (material.material_type == MaterialType::Diffuse) {
                                    brdf = materialColor / PI;
                                }
                                else if (material.material_type == MaterialType::Microfacet) {
                                    brdf = CookTorrance::BRDF(-path.ray.direction, normal, light_dir, materialColor, roughness, metallic);
                                }
                            #endif

                            int light_id = direct_light_intersection.materialId;

                            #if UBER_SHADER
                                glm::vec3 light_radiance = get_emission(materials[light_id], direct_light_intersection.uvs, textures);
                            #else
                                glm::vec3 light_color = materials[light_id].color;

                                if (materials[light_id].albedo_tex >= 0) {
                                    glm::vec2 dl_uv = direct_light_intersection.uvs;

                                    dl_uv *= materials[light_id].albedo_tex_transform.scale;

                                    if(materials[light_id].albedo_tex_transform.rotation) {
                                        float c = cosf(materials[light_id].albedo_tex_transform.rotation);
                                        float s = sinf(materials[light_id].albedo_tex_transform.rotation);

                                        dl_uv = glm::vec2 (
                                            c * dl_uv.x - s * dl_uv.y,
                                            s * dl_uv.x + c * dl_uv.y
                                        );
                                    }

                                    dl_uv += materials[light_id].albedo_tex_transform.offset;

                                    float4 tex = tex2D<float4>(textures[materials[light_id].albedo_tex].tex, dl_uv.x, dl_uv.y);
                                    light_color = glm::pow(glm::vec3(tex.x, tex.y, tex.z), glm::vec3(2.2f));
                                }

                                glm::vec3 light_radiance = materials[light_id].emittance * light_color;
                            #endif
                            glm::vec3 contribution = (light_radiance * brdf * cosThetaSurface) / pdf_dl_sa;
                            path.color += path.throughput * contribution * mis_weight;
                        }
                    }
                }
            }
            #if UBER_SHADER
            else if (glm::length(emission) > EPSILON) {
                path.color += path.throughput * emission;
                path.kill = true;
            }
            #else
            else if (material.material_type == MaterialType::Emissive) {
                path.color += path.throughput * material.emittance * materialColor;
                path.kill = true;
            }
            #endif

            if (!path.kill) {
                update_throughput_materials(
                    material,
                    path,
                    idx,
                    num_paths,
                    iter,
                    depth,
                    rng,
                    materialColor,
                    normal,
                    roughness,
                    metallic,
                    is_specular
                );
            }
        }
    }   
    else if (!path.kill && has_exr) {
        // hdri
        glm::vec3 d = glm::normalize(path.ray.direction);
        float phi = atan2f(d.z, d.x);
        float theta = glm::acos(glm::clamp(d.y, -1.0f, 1.0f));
        
        float u = (phi + M_PI) * (1.0f / (2.0f * M_PI));
        float v = theta * (1.0f / M_PI);

        float4 env = tex2D<float4>(exr, u, v);
        
        glm::vec3 env_radiance = glm::vec3(env.x, env.y, env.z) * DEV_OPTIONS.envmap_intensity;

        float mis_weight = 1.0f;
        if (DEV_OPTIONS.environment_map_importance_sampling && depth > 1 && !path.last_bounce_was_specular) {
            float pdf_bsdf = path.last_pdf;
            float pdf_env  = envmap_pdf(d, marginal_cdf, conditional_cdfs, exr_width, exr_height);
            mis_weight = (pdf_bsdf * pdf_bsdf) / (pdf_bsdf * pdf_bsdf + pdf_env * pdf_env);
        }

        path.color += path.throughput * env_radiance * mis_weight;
    }
    
    if (intersection.t == -1.0f) {
        path.kill = true;
    }
    else {
        Ray& ray = path.ray;
        glm::vec3 hit_point = getPointOnRay(ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        // Guard for paths
        float sample_len2 = glm::dot(path.sample_dir, path.sample_dir);
        if (!(sample_len2 > 1e-12f) || !isfinite(sample_len2)) {
            path.kill = true;
            return;
        }

        ray.direction = path.sample_dir;

        if (glm::dot(ray.direction, normal) > 0.0f) {
            ray.origin = hit_point + (normal * EPSILON);
        }
        else {
            ray.origin = hit_point - (normal * EPSILON);
        }   
    }
}
