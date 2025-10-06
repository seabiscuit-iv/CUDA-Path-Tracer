#ifndef COOK_TORRANCE
#define COOK_TORRANCE

#include "common.cu"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#define INV_PI 0.3183098f

#define CLAMP_POS(x) glm::max(x, 0.0f)

// This is a workaround, and is not physically accurate. This should be replaced with direct light sampling and MIS
#define MICROFACET_REMOVE_FIREFLIES 0

namespace CookTorrance {

    __device__ float D_TrowbridgeReitz(glm::vec3 h, glm::vec3 n, float alpha) {
        float alpha_sq = alpha * alpha;

        float numerator = alpha_sq;

        float n_dot_h_sq = glm::pow( CLAMP_POS(glm::dot(n, h)) , 2.0f);
        float denom_component = ( n_dot_h_sq * (alpha_sq - 1.0f) ) + 1.0f;

        float denominator = denom_component * denom_component * glm::pi<float>();

        return numerator / glm::max(denominator, 0.0001f);
    }

    __device__ glm::vec3 F_SchlickApprox(float v_dot_h, const glm::vec3& f0) {
        return f0 + ( (glm::vec3(1.0f) - f0) * glm::pow( ( 1.0f - v_dot_h ), 5.0f ) );
    }

    __device__ float Smith_GGX(glm::vec3 w, glm::vec3 n, float alpha) {
        float alpha_sq = alpha * alpha;
        float n_dot_w = CLAMP_POS(glm::dot(n, w));

        float numerator = 2.0f * n_dot_w;

        float under_sqrt = alpha_sq + (1.0f - alpha_sq) * (glm::pow(n_dot_w, 2.0f));
        float denominator = n_dot_w + glm::sqrt(under_sqrt);

        return numerator / glm::max(denominator, 0.0001f);
    }

    __device__ float Smith_G(glm::vec3 v, glm::vec3 l, glm::vec3 n, float alpha) {
        return Smith_GGX(v, n, alpha) * Smith_GGX(l, n, alpha);
    }
    
    __device__ glm::vec3 BRDF(glm::vec3 v, glm::vec3 n, glm::vec3 l, glm::vec3 albedo, float roughness, float metallic) {
        glm::vec3 h = glm::normalize(v + l);

        roughness = glm::clamp(roughness, 0.05f, 1.0f);
        float alpha = roughness * roughness;

        glm::vec3 dielectricF0 = glm::vec3(0.04f);
        glm::vec3 F0 = glm::mix(dielectricF0, albedo, metallic);

        float D = D_TrowbridgeReitz(h, n, alpha);
        glm::vec3 F = F_SchlickApprox( CLAMP_POS(glm::dot(v, h)), F0); // this should be changed to respect IOR
        float G = Smith_G(v, l, n, alpha);

        glm::vec3 numerator = D * F * G;
        float denominator = 4 * CLAMP_POS(glm::dot(n, v)) * CLAMP_POS(glm::dot(n, l));

        glm::vec3 specular = numerator / glm::max(denominator, 0.0001f);

        glm::vec3 nonSpecular = glm::vec3(1.0f) - F; // Use the same F calculated for specular
        glm::vec3 k_S = F; // Specular ratio
        glm::vec3 k_D = (1.0f - metallic) * nonSpecular; // Diffuse ratio, must be 0 for full metallic

        glm::vec3 diffuse = k_D * albedo * INV_PI;

        return diffuse + specular;
    }

    __host__ __device__ glm::vec3 sphericalToCartesian(glm::vec3 spherical) {
        float r = spherical.x;
        float theta = spherical.y;
        float phi = spherical.z;

        float x = r * glm::sin(theta) * glm::cos(phi);
        float y = r * glm::sin(theta) * glm::sin(phi);
        float z = r * glm::cos(theta);

        return glm::vec3(x, y, z);
    } 

    __device__ void sampleGGX(PathSegment &path, int idx, int iter, int depth, glm::vec3 wo, glm::vec3 n, float roughness) {
        CREATE_RANDOM_ENGINE(iter, idx, depth, u01, rng);

        float x1 = u01(rng);
        float x2 = u01(rng);

        roughness = glm::clamp(roughness, 0.05f, 1.0f);
        float alpha = roughness * roughness;

        float under_atan = (alpha * glm::sqrt(x1)) / glm::sqrt(1.0f - x1); 
        float theta_h = glm::atan(under_atan);

        float phi_h = 2.0f * glm::pi<float>() * x2;

        glm::vec3 h = glm::normalize(sphericalToCartesian(glm::vec3(1.0, theta_h, phi_h)));

        glm::vec3 t = glm::normalize( abs(n.z) < 0.999f ? glm::cross(glm::vec3(0,0,1), n) : glm::cross(glm::vec3(1,0,0), n) );
        glm::vec3 b = glm::cross(n, t);

        glm::vec3 h_world = h.x * t + h.y * b + h.z * n;

        glm::vec3 wi = glm::reflect( -wo, h_world ); //wi and wo both point out of the intersection

        if (glm::dot(wi, n) < 0.0f) {
            wi = -wi;
        }

        path.sample_dir = wi;
    }

    __device__ void sampleCookTorrance(PathSegment &path, Material &material, int idx, int iter, int depth, glm::vec3 wo, glm::vec3 n, float roughness) {
        CREATE_RANDOM_ENGINE(iter, idx, depth, u01, rng);
        float r = u01(rng);

        glm::vec3 dielectricF0 = glm::vec3(0.04f);
        glm::vec3 F0 = glm::mix(dielectricF0, material.color, material.metallic);
        float probSpecular = glm::clamp(F_SchlickApprox(glm::dot(wo, n), glm::vec3(F0)).r, 0.05f, 0.95f);

        if (r <= probSpecular) {
            sampleGGX(path, idx, iter, depth, wo, n, roughness);
        } else {
            glm::vec3 wo = -path.ray.direction;
            glm::vec3 wi;
            wi = calculateRandomDirectionInHemisphere(n, rng);
            path.sample_dir = wi;
        }
    }

    __device__ float PDF_GGX( glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness) {
        glm::vec3 h = glm::normalize(wo + wi);

        roughness = glm::clamp(roughness, 0.05f, 1.0f);
        float alpha = roughness * roughness;

        float p_h = D_TrowbridgeReitz(h, n, alpha) * glm::dot(n, h);

        float p_wi = p_h / glm::max( (4 * glm::abs(glm::dot(wo, h))), 0.0001f);

        return p_wi;
    }  

    
    __device__ float PDF(const Material &material, glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness) {
        float pdfDiffuse  = max(0.0f, glm::dot(wi, n)) * INV_PI;
        float pdfSpecular = PDF_GGX(wo, wi, n, roughness);

        glm::vec3 dielectricF0 = glm::vec3(0.04f);
        glm::vec3 F0 = glm::mix(dielectricF0, material.color, material.metallic);
        float probSpecular = glm::clamp(F_SchlickApprox(glm::dot(wo, n), glm::vec3(F0)).r, 0.05f, 0.95f);

        float pdf = (1.0f - probSpecular) * pdfDiffuse + probSpecular * pdfSpecular;
        return pdf;
    }

    
    __device__ void shadePathCookTorrance(
        ShadeableIntersection &intersection,
        PathSegment &path,
        const Material &material
    )
    {
        glm::vec3 albedo = material.color;
        glm::vec3 wo = -path.ray.direction;
        glm::vec3 wi = path.sample_dir;
        glm::vec3 normal = intersection.surfaceNormal;

        float roughness = material.roughness;
        float metallic = material.metallic;

        glm::vec3 brdf = BRDF(wo, normal, wi, albedo, roughness, metallic);

        #if MICROFACET_REMOVE_FIREFLIES
            brdf = glm::clamp(brdf, glm::vec3(0.0), glm::vec3(1.0));
        #endif // REMOVE_FIREFLIES

        float absdot = max(0.0f, glm::dot(wi, normal));
        float pdf = PDF(material, wo, wi, normal, roughness);

        // MICROFACET PDF CLAMP, THIS IS NECESSARY TO REMOVE FIREFLIES
        path.throughput *= brdf * absdot / glm::max(pdf, 0.025f);
    }   

}

#endif // COOK_TORRANCE