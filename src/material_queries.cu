#include "material_queries.h"

__device__ glm::vec3 get_albedo(const Material& material, glm::vec2 uv, const TextureData* textures) {
    if(material.albedo_tex >= 0) {
        uv *= material.albedo_tex_transform.scale;
        
        if(material.albedo_tex_transform.rotation) {
            float c = cosf(material.albedo_tex_transform.rotation);
            float s = sinf(material.albedo_tex_transform.rotation);

            uv = glm::vec2 (
                c * uv.x - s * uv.y,
                s * uv.x + c * uv.y
            );
        }

        uv += material.albedo_tex_transform.offset;

        float4 tex = tex2D<float4>(textures[material.albedo_tex].tex, uv.x, uv.y);
        return glm::pow(glm::vec3(tex.x, tex.y, tex.z), glm::vec3(2.2f));
    }
    else {
        return material.color;
    }
}


__device__ glm::vec3 get_normal(const Material& material, const TextureData* textures, const ShadeableIntersection& intersection, glm::vec3* out_normal_map) {
    if (material.normal_tex >= 0) {
        glm::vec2 uv = intersection.uvs;

        uv *= material.normal_tex_transform.scale;
        
        if(material.normal_tex_transform.rotation) {
            float c = cosf(material.normal_tex_transform.rotation);
            float s = sinf(material.normal_tex_transform.rotation);

            uv = glm::vec2 (
                c * uv.x - s * uv.y,
                s * uv.x + c * uv.y
            );
        }

        uv += material.normal_tex_transform.offset;


        float4 tex = tex2D<float4>(textures[material.normal_tex].tex, uv.x, uv.y);
        glm::vec3 local_normal = glm::vec3(tex.x, tex.y, tex.z);

        *out_normal_map = local_normal;

        local_normal.x = local_normal.r * 2.0f - 1.0f;
        local_normal.y = local_normal.g * 2.0f - 1.0f;
        local_normal.z = local_normal.b * 2.0f - 1.0f;

        glm::vec3 bitangent = glm::normalize(glm::cross(intersection.surfaceTangent, intersection.surfaceNormal));

        glm::mat3 TBN = glm::mat3(intersection.surfaceTangent, bitangent, intersection.surfaceNormal);

        return glm::normalize(TBN * local_normal);
    }
    else {
        return intersection.surfaceNormal;
    }
}