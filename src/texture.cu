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
        std::vector<glm::vec4>& src = host_texture_data[i];

        std::vector<float4> tmp(src.size());
        for (size_t j = 0; j < src.size(); j++) {
            tmp[j] = make_float4(
                src[j].x,
                src[j].y,
                src[j].z,
                src[j].w
            );
        }

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();

        cudaArray_t cuArray;
        cudaMallocArray(
            &cuArray,
            &desc,
            tex.width,
            tex.height
        );

        cudaMemcpy2DToArray(
            cuArray,
            0, 0,
            tmp.data(),
            tex.width * sizeof(float4),
            tex.width * sizeof(float4),
            tex.height,
            cudaMemcpyHostToDevice
        );

        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;    
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc{};  
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode     = cudaFilterModeLinear;
        texDesc.readMode       = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(
            &tex.tex,
            &resDesc,
            &texDesc,
            nullptr
        );
        cuda_arrays.push_back(cuArray);
    }

    cudaMalloc(&dev_textures, sizeof(TextureData) * host_textures.size());
    cudaMemcpy(dev_textures, host_textures.data(), sizeof(TextureData) * host_textures.size(), cudaMemcpyHostToDevice);
}

void TextureHandler::free() {
    for(auto& tex : host_textures) {
        cudaDestroyTextureObject(tex.tex);
    }

    for (auto& arr : cuda_arrays) {
        cudaFreeArray(arr);
    }

    cudaFree(dev_textures);

    host_textures.clear();
    host_texture_data.clear();
    cuda_arrays.clear();
}