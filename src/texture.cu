#include "texture.h"
#include "cuda_runtime.h"

void TextureHandler::load_texture(std::vector<glm::vec4>& data, int width, int height) {
    TextureData tex;
    tex.width = width;
    tex.height = height;
    host_textures.push_back(tex);
    host_texture_data.push_back(data);
}

void TextureHandler::load_textures_on_device() {
    for (int i = 0; i < host_textures.size(); i++) {
        TextureData& tex = host_textures[i];
        std::vector<glm::vec4>& data = host_texture_data[i];

        cudaMalloc(&tex.data, data.size() * sizeof(glm::vec4));
        cudaMemcpy(tex.data, data.data(), tex.width * tex.height * sizeof(glm::vec4), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&dev_textures, sizeof(TextureData) * host_textures.size());
    cudaMemcpy(dev_textures, host_textures.data(), sizeof(TextureData) * host_textures.size(), cudaMemcpyHostToDevice);
}

void TextureHandler::free() {
    cudaFree(dev_textures);
    for(auto& tex : host_textures) {
        cudaFree(tex.data);
    }

    host_textures.clear();
}