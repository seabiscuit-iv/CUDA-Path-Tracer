#pragma once

#include <glm/glm.hpp>
#include <vector>

struct TextureData {
    int width;
    int height;
    glm::vec4* data;
};

struct TextureHandler {
public:
    TextureHandler(const TextureHandler&) = delete;
    void operator=(const TextureHandler&) = delete;

    static TextureHandler& get() {
        static TextureHandler instance;
        return instance;
    }

    void load_texture(std::vector<glm::vec4>& data, int width, int height);

    void load_textures_on_device();

    void free();

private:
    TextureHandler() {}

    std::vector<TextureData> host_textures;
    TextureData* dev_textures;

    std::vector<std::vector<glm::vec4>> host_texture_data;
};