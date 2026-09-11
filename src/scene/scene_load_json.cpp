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

#include "scene/cube_mesh.h"

#include "json.hpp"
#include "tinyobj/tiny_obj_loader.h"

using json = nlohmann::json;

#if LOAD_FROM_JSON
void Scene::loadFromJSON(const std::string& jsonName, std::string exr_path)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            #if !UBER_SHADER
                newMaterial.material_type = MaterialType::Diffuse;
            #endif
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            #if UBER_SHADER
                newMaterial.emission.emission_color = glm::vec3(col[0], col[1], col[2]);
                newMaterial.emission.emission_strength = p["EMITTANCE"];
            #else
                newMaterial.emittance = p["EMITTANCE"];
                newMaterial.material_type = MaterialType::Emissive;
            #endif
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            #if !UBER_SHADER
                newMaterial.material_type = MaterialType::Specular;
            #endif
        }
        else if (p["TYPE"] == "Microfacet")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.material_type = MaterialType::Microfacet;
            const auto& roughness = p["ROUGHNESS"];
            const auto& metallic = p["METALLIC"];
            newMaterial.roughness = roughness;
            newMaterial.metallic = metallic;
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            #if !UBER_SHADER
                newMaterial.material_type = MaterialType::Glass;
            #endif
        }
        #if UBER_SHADER
            newMaterial.material_type = MaterialType::Microfacet;
        #endif
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
        material_names.push_back(name);
    }
    const auto& objectsData = data["Objects"];
    
    int count = 0;
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = GeomType::MESH;  
            newGeom.mesh.make_mesh_host(CUBE_VERTICES, CUBE_INDICES, CUBE_NORMALS, CUBE_NORMAL_INDICES, {}, {});
            newGeom.mesh.label = "Cube";
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh") 
        {
            newGeom.type = GeomType::MESH;
            const auto& verts = p["VERTICES"];

            std::vector<glm::vec3> hostVerts;
            std::vector<int> hostIndices;
            for (size_t i = 0; i < verts.size(); i++) {
                const auto& v = verts[i];
                hostVerts.push_back(glm::vec3(v[0], v[1], v[2]));
                hostIndices.push_back(int(i));
            }

            newGeom.mesh.make_mesh_host(hostVerts, hostIndices, std::vector<glm::vec3>(), std::vector<int>(), {}, {});
            newGeom.mesh.label = fmt::format("__OBJECT{}__", count);
        }
        else if (type == "obj")
        {
            newGeom.type = GeomType::MESH;
            const auto& f_n = p["FILE"];

            std::string file_name(f_n);

            tinyobj::attrib_t attributes;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string err = "";
            tinyobj::LoadObj(&attributes, &shapes, &materials, &err, file_name.c_str(), nullptr, true);

            if (err != "" ) {
                printf("tinyobj Model Loading Error: %s\n", err.c_str());
            }

            std::vector<glm::vec3> hostVerts;
            std::vector<int> hostIndices;
            for(const tinyobj::shape_t &shape : shapes) {
                for (int i = 0; i < attributes.vertices.size(); i += 3) {
                    float v1 = attributes.vertices[i];
                    float v2 = attributes.vertices[i + 1];
                    float v3 = attributes.vertices[i + 2];
                    hostVerts.push_back(glm::vec3(v1, v2, v3));
                }
                for(int i = 0; i < shape.mesh.indices.size(); i++) {
                    int vi = shape.mesh.indices[i].vertex_index;
                    hostIndices.push_back(vi);
                }
            }

            std::vector<glm::vec3> hostNormals;
            std::vector<int> hostNormalIndices;
            for(const tinyobj::shape_t &shape : shapes) {
                for (int i = 0; i < attributes.normals.size(); i += 3) {
                    float v1 = attributes.normals[i];
                    float v2 = attributes.normals[i + 1];
                    float v3 = attributes.normals[i + 2];
                    hostNormals.push_back(glm::vec3(v1, v2, v3));
                }
                for(int i = 0; i < shape.mesh.indices.size(); i++) {
                    int ni = shape.mesh.indices[i].normal_index;
                    hostNormalIndices.push_back(ni);
                }
            }
            
            newGeom.mesh.make_mesh_host(hostVerts, hostIndices, hostNormals, hostNormalIndices, {}, {});
            newGeom.mesh.label = file_name;
        }
        else    
        {
            printf("ERROR: Unrecognized geometry type %s\n", type.type_name());
            exit(1);
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        count++;
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    camera.view  = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up    = glm::normalize(glm::cross(camera.right, camera.view));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    if (!exr_path.empty()) {
        float* exr;
        const char* exr_err = NULL;

        int exr_ret = LoadEXR(&exr, &exr_width, &exr_height, exr_path.c_str(), &exr_err);

        if (exr_ret != TINYEXR_SUCCESS) {
            if (exr_err) {
                fprintf(stderr, "ERR : %s\n", exr_err);
                FreeEXRErrorMessage(exr_err);
            }
            exit(1);
        } else {
            fmt::println("Exr loading success: {} x {}", exr_width, exr_height);

            exr_data.resize(exr_width * exr_height);
            memcpy(exr_data.data(), exr, 4 * exr_width * exr_height * sizeof(float));

            free(exr);
        }
    }

    precompute_emissive_mesh_area();
    precompute_hdri_emission();
}
#endif
