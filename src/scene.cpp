#include "scene.h"

#include "utilities.h"

#include <cuda.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "json.hpp"

#include "tinyobj/tiny_obj_loader.h"

#include "tinygltf/tiny_gltf.h"
#include "tinyexr/tinyexr.h"

#include <fstream>
#include <iostream>
#include <string>
#include <fmt/format.h>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename, const char* env_map_path)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    if (env_map_path) {
        printf("Using environment map: %s\n", env_map_path);
    }
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename, env_map_path ? std::string(env_map_path) : std::string());
        return;
    }
    else if (ext == ".glb") {
        loadFromGLTF(filename, env_map_path ? std::string(env_map_path) : std::string());
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

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
            newMaterial.material_type = MaterialType::Diffuse;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.material_type = MaterialType::Emissive;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.material_type = MaterialType::Specular;
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
            newMaterial.material_type = MaterialType::Glass;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
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
            newGeom.mesh.make_mesh_host(CUBE_VERTICES, CUBE_INDICES, CUBE_NORMALS, CUBE_NORMAL_INDICES);
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

            newGeom.mesh.make_mesh_host(hostVerts, hostIndices, std::vector<glm::vec3>(), std::vector<int>());
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
            
            newGeom.mesh.make_mesh_host(hostVerts, hostIndices, hostNormals, hostNormalIndices);
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
}



void Scene::loadFromGLTF(const std::string& gltfName, std::string exr_path) {
    fmt::println("Loading {} as .glb file", gltfName);

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    // bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename); // for binary glTF(.glb)

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF: %s\n", gltfName.c_str());
        exit(1);
    }

    Material defaultMat{};
    defaultMat.color = glm::vec3(1.0f);
    defaultMat.material_type = MaterialType::Emissive;
    defaultMat.emittance = 10.0f;
    materials.push_back(defaultMat);


    for (auto& mat : model.materials) {
        Material newMaterial{};
        if (mat.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            newMaterial.color = glm::vec3(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2]
            );
        }
        newMaterial.metallic = mat.pbrMetallicRoughness.metallicFactor;
        newMaterial.roughness = mat.pbrMetallicRoughness.roughnessFactor;


        float emissive_strength = 1.0f;
        if (mat.extensions.find("KHR_materials_emissive_strength") != mat.extensions.end()) {
            const auto& ext = mat.extensions.at("KHR_materials_emissive_strength");

            if (ext.Has("emissiveStrength")) {
                emissive_strength = static_cast<float>(ext.Get("emissiveStrength").GetNumberAsDouble());
            }
        }
        newMaterial.emittance = emissive_strength * glm::length(glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]));

        if (newMaterial.emittance > 0.01f) {
            newMaterial.material_type = MaterialType::Emissive;
            newMaterial.color = glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
        }
        else if (isGlass(mat)) {
            newMaterial.material_type = MaterialType::Glass;
            newMaterial.alpha = static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[3]);
        }
        else if (newMaterial.metallic < 0.01f && newMaterial.roughness > 0.99f) {
            newMaterial.material_type = MaterialType::Diffuse;
        }
        else {
            newMaterial.material_type = MaterialType::Microfacet;
        }

        materials.push_back(newMaterial);
        // fmt::println("New Material {} with RGB {} of type {}", mat.name, glm::to_string(newMaterial.color), (int)newMaterial.material_type);
    }

    int camera_node = -1;
    glm::mat4 camera_transform;

    std::function<void(int, glm::mat4)> processNode;
    processNode = [&](int node_idx, glm::mat4 parent_transform)
    {
        const auto& node = model.nodes[node_idx];
        glm::mat4 node_transform = glm::mat4(1.0f);

        if (!node.matrix.empty()) {
            node_transform = glm::make_mat4(node.matrix.data());
        }
        else {
            if (!node.translation.empty()) {
                node_transform = glm::translate(
                    node_transform, 
                    glm::vec3(
                        node.translation[0], 
                        node.translation[1], 
                        node.translation[2]
                    )
                );
            }

            if (!node.rotation.empty()) {
                glm::quat q(
                    node.rotation[3], 
                    node.rotation[0], 
                    node.rotation[1], 
                    node.rotation[2]
                );
                node_transform *= glm::mat4_cast(q);
            }

            if (!node.scale.empty()) {
                node_transform = glm::scale(
                    node_transform,
                    glm::vec3(
                        node.scale[0],
                        node.scale[1],
                        node.scale[2]
                    )
                );
            }
        }

        glm::mat4 global_transform = parent_transform * node_transform;

        if (node.mesh >= 0) {
            const auto& mesh = model.meshes[node.mesh];
            for (const auto& prim : mesh.primitives) {
                if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                    fmt::println("WARNING: Mesh {} attempted to create a non-triangle mode primitive", mesh.name);
                    continue;
                }

                Geom new_geom{};

                new_geom.type = GeomType::MESH;

                auto it = prim.attributes.find("POSITION");
                if (it == prim.attributes.end()) { 
                    continue;
                }
                const auto& pos_accessor = model.accessors[it->second];

                assert(pos_accessor.type == TINYGLTF_TYPE_VEC3);
                assert(pos_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                const auto& pos_buffer_view = model.bufferViews[pos_accessor.bufferView];
                const auto& buffer = model.buffers[pos_buffer_view.buffer];

                std::vector<glm::vec3> vertices(pos_accessor.count);

                size_t stride = pos_buffer_view.byteStride
                    ? pos_buffer_view.byteStride
                    : sizeof(float) * 3;

                const uint8_t* base =
                    buffer.data.data() +
                    pos_buffer_view.byteOffset +
                    pos_accessor.byteOffset;

                for (size_t i = 0; i < pos_accessor.count; i++) {
                    const float* p = reinterpret_cast<const float*>(base + i * stride);
                    vertices[i] = { p[0], p[1], p[2] };
                }

                std::vector<glm::vec3> normals {};
                auto it_norm = prim.attributes.find("NORMAL");

                if (it_norm != prim.attributes.end()) {
                    const auto& norm_accessor = model.accessors[it_norm->second];
                    const auto& norm_buffer_view = model.bufferViews[norm_accessor.bufferView];
                    const auto& norm_buffer = model.buffers[norm_buffer_view.buffer];

                    normals.resize(norm_accessor.count);

                    size_t norm_stride = norm_buffer_view.byteStride 
                        ? norm_buffer_view.byteStride 
                        : sizeof(float) * 3;

                    const uint8_t* norm_base = 
                        norm_buffer.data.data() + 
                        norm_buffer_view.byteOffset + 
                        norm_accessor.byteOffset;

                    for (size_t i = 0; i < norm_accessor.count; i++) {
                        const float* n = reinterpret_cast<const float*>(norm_base + i * norm_stride);
                        normals[i] = glm::normalize(glm::vec3{ n[0], n[1], n[2] });
                    }
                }

                std::vector<int> indices;
                if (prim.indices >= 0) {
                    const auto& idx_accessor = model.accessors[prim.indices];
                    const auto& idx_buffer_view = model.bufferViews[idx_accessor.bufferView];
                    const auto& idx_buffer = model.buffers[idx_buffer_view.buffer];
                    const void* idx_data = &idx_buffer.data[idx_buffer_view.byteOffset + idx_accessor.byteOffset];

                    indices.resize(idx_accessor.count);
                    if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* buf = static_cast<const uint16_t*>(idx_data);
                        for (size_t i = 0; i < idx_accessor.count; ++i) {
                            indices[i] = buf[i];
                        }
                    } 
                    else if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        const uint32_t* buf = static_cast<const uint32_t*>(idx_data);
                        for (size_t i = 0; i < idx_accessor.count; ++i) {
                            indices[i] = buf[i];
                        }
                    }
                    else if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t* buf = static_cast<const uint8_t*>(idx_data);
                        for (size_t i = 0; i < idx_accessor.count; ++i) { 
                            indices[i] = buf[i];
                        }
                    }
                } 
                else {
                    indices.resize(vertices.size());
                    std::iota(indices.begin(), indices.end(), 0);
                }

                new_geom.mesh.make_mesh_host(vertices, indices, normals, indices);
                new_geom.mesh.label = mesh.name;
                new_geom.transform = global_transform;
                new_geom.inverseTransform = glm::inverse(global_transform);
                new_geom.invTranspose = glm::inverseTranspose(global_transform);

                new_geom.materialid = (prim.material >= 0) ? prim.material + 1 : 0; // default material id
                geoms.push_back(new_geom);
            }
        }

        if (node.camera >= 0) {
            if (camera_node != -1) {
                fmt::println("Multiple cameras defined in GLTF, exiting");
                exit(1);
            }

            camera_node = node_idx;
            camera_transform = global_transform;
        }

        for (auto child : node.children) {
            processNode(child, global_transform);
        }
    };

    int sceneIndex = (model.defaultScene >= 0) ? model.defaultScene : 0;
    for (size_t i = 0; i < model.scenes[sceneIndex].nodes.size(); i++) {
        processNode(model.scenes[sceneIndex].nodes[i], glm::mat4(1.0f));
    }

    if (camera_node < 0) {
        fmt::println("No camera found in glTF");
        exit(1);
    }

    tinygltf::Camera& gltf_camera = model.cameras[model.nodes[camera_node].camera];
    fmt::println("Found camera {} at node {}", gltf_camera.name, camera_node);

    float aspect = static_cast<float>(gltf_camera.perspective.aspectRatio);

    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.y = 1000; // this should be customized
    camera.resolution.x = aspect * camera.resolution.y; // this should be customized
    float fovy = glm::degrees(gltf_camera.perspective.yfov);
    state.iterations = 5000; // this should be customized
    state.traceDepth = 8; // this should be customized
    state.imageName = model.scenes[sceneIndex].name.empty() ? "GLTF_DEFAULT_NAME" : model.scenes[sceneIndex].name;

    glm::vec3 eye = glm::vec3(camera_transform[3]);

    camera.position = eye;
    
    camera.view  = -glm::normalize(glm::vec3(camera_transform[2]));
    camera.right = glm::normalize(glm::vec3(camera_transform[0])); 
    camera.up    = glm::normalize(glm::vec3(camera_transform[1]));

    camera.lookAt = camera.position + camera.view;

    // //calculate fov based on resolution
    float yscaled = tan(0.5f * fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (2.0f * atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // //set up render camera stuff
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
}