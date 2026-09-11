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
        #if LOAD_FROM_JSON
            loadFromJSON(filename, env_map_path ? std::string(env_map_path) : std::string());
        #else
            fmt::println("JSON Model Loading has been deprecated (config.h)");
            exit(1);
        #endif
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
