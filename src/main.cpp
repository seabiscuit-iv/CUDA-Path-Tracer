#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "myoptix.h"
#include "config.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;
static glm::vec3 refUp;

float zoom, theta, phi;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo = 0;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

void terminateHandler() {
    if (auto ex = std::current_exception()) {
        try {
            std::rethrow_exception(ex);
        } catch (const std::exception& e) {
            std::cerr << "Uncaught exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Uncaught non-standard exception" << std::endl;
        }
    } else {
        std::cerr << "Terminate called without active exception" << std::endl;
    }
    std::abort();
}

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    init_optix();

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    bool changed = false;

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    
    #if !OPTIX
        changed |= ImGui::Checkbox("Debug BVH", &PathTracerOptions::Get()->debug_bvh);
    #endif

    changed |= ImGui::Checkbox("Material Debug Mode", &PathTracerOptions::Get()->material_debug_mode);
    
    ImGui::End();

    if (changed) {
        camchanged = true;
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    std::set_terminate(terminateHandler);

    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    // Ensure normalized view direction
    glm::vec3 v = glm::normalize(cam.view);
    // Horizontal angle (yaw) around Y axis
    // 0 when looking down -Z
    phi = atan2(v.x, -v.z);
    // Vertical angle (pitch) from +Y
    // 0 when looking straight up, π when looking straight down
    theta = acos(glm::clamp(v.y, -1.0f, 1.0f));
    zoom = glm::length(cam.position - cam.lookAt);
    refUp = glm::normalize(glm::vec3(0.0, 1.0, 0.0));

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    for(Geom &g : scene->geoms) {
        if (g.type == GeomType::MESH && g.mesh.h_valid) {
            bool copied = false;
            for(Geom &s : scene->geoms) {
                if (s.type == GeomType::MESH && s.mesh.h_valid && s.mesh.d_valid && s.mesh.label == g.mesh.label && s.materialid == g.materialid) {
                    g.mesh.make_mesh_device_copy(s.mesh);
                    copied = true;
                    break;
                }
            }
            
            if (copied) {
                continue;
            }

            g.mesh.make_mesh_device();
        }
    }

    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    compile_pathtracing_optix_module(module, pipeline_compile_options); 

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hit_prog_group = nullptr;
    create_optix_program_groups(module, raygen_prog_group, miss_prog_group, hit_prog_group);

    OptixPipeline optix_pipeline;
    initialize_optix_pipeline(raygen_prog_group, miss_prog_group, hit_prog_group, pipeline_compile_options, optix_pipeline);

    OptixShaderBindingTable sbt = {};
    create_optix_sbt(sbt, raygen_prog_group, miss_prog_group, hit_prog_group);

    std::vector<OptixInstance> optix_instances;
    int id = 0;
    for(Geom &g : scene->geoms) {
        if (g.type == GeomType::MESH && g.mesh.d_valid) {
            // Create a single IAS from all GAS
            OptixInstance inst = {};

            float transform[12] = {
                g.transform[0][0], g.transform[1][0], g.transform[2][0], g.transform[3][0],
                g.transform[0][1], g.transform[1][1], g.transform[2][1], g.transform[3][1],
                g.transform[0][2], g.transform[1][2], g.transform[2][2], g.transform[3][2],
            };  

            memcpy(inst.transform, transform, sizeof(float) * 12);

            inst.instanceId = id;
            // inst.sbtOffset = id * RAY_TYPE_COUNT;
            inst.sbtOffset = 0;
            inst.visibilityMask = 255;
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = g.mesh.as_handle;

            optix_instances.push_back(inst);

            id++;
        }
    }

    CUdeviceptr d_optix_instances;
    OptixTraversableHandle ias_handle;

    create_ias(optix_instances, d_optix_instances, ias_handle);

    scene->optix_pipeline = optix_pipeline;
    scene->ias_handle = ias_handle;
    scene->optix_sbt = sbt;

    // GLFW main loop
    mainLoop();

    for(Geom g : scene->geoms) {
        if (g.type == GeomType::MESH && g.mesh.d_valid) {
            g.mesh.delete_mesh_device();
        }
    }

    return 0;
}

glm::vec3 ACESFilmHost(glm::vec3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}


void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index] / samples;

            //reinhard op
            // pix = pix / (pix + glm::vec3(1.0f));
            pix = ACESFilmHost(pix);

            //gamma correction
            pix = glm::pow(pix, glm::vec3(0.45f));

            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << "img/" << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;

        glm::vec3 focus_point = cam.lookAt;
        glm::vec3 sphericals;

        cam.view = cam.lookAt - cam.position;

        sphericals.x = -zoom * sin(phi) * sin(theta);
        sphericals.y = -zoom * cos(theta);
        sphericals.z = zoom * cos(phi) * sin(theta);

        // fmt::println("OG: {}", glm::to_string(cam.view));
        // fmt::println("NEW: {}", glm::to_string(-glm::normalize(sphericals)));

        cam.view = -glm::normalize(sphericals);
        glm::vec3 v = cam.view;
        glm::vec3 u = refUp - v * glm::dot(refUp, v);
        if (glm::pow(glm::length(u), 2.0f) < 1e-6f) {
            u = glm::vec3(0, 1, 0);
        }
        u = glm::normalize(u);
        glm::vec3 r = glm::normalize(glm::cross(v, u));
        u = glm::cross(r, v);
        cam.up    = u;
        cam.right = r;

        cam.position = cam.lookAt - cam.view * zoom;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

bool shiftPressed(GLFWwindow* window)
{
    return glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                saveImage();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX && ypos == lastY)
        return;

    double dx = xpos - lastX;
    double dy = ypos - lastY;

    if (MouseOverImGuiWindow())
        goto end;

    // SHIFT + LEFT DRAG → PAN
    if (leftMousePressed && shiftPressed(window))
    {
        Camera& cam = renderState->camera;

        float panSpeed = zoom * 0.0015f;

        glm::vec3 right = cam.right;
        glm::vec3 up    = cam.up;

        cam.lookAt -= right * float(dx) * panSpeed;
        cam.lookAt += up    * float(dy) * panSpeed;

        cam.position -= right * float(dx) * panSpeed;
        cam.position += up    * float(dy) * panSpeed;

        camchanged = true;
    }
    // LEFT DRAG → ORBIT
    else if (leftMousePressed)
    {
        phi   -= dx / width;
        theta -= dy / height;

        theta = glm::clamp(theta, 0.001f, PI - 0.001f);
        camchanged = true;
    }
    // RIGHT DRAG → ZOOM (optional)
    else if (rightMousePressed)
    {
        zoom *= std::exp(float(dy) * 0.002f);
        zoom = glm::clamp(zoom, 0.1f, 1000.0f);
        camchanged = true;
    }
    // MIDDLE DRAG → PAN (legacy support)
    else if (middleMousePressed)
    {
        Camera& cam = renderState->camera;

        float panSpeed = zoom * 0.0015f;

        glm::vec3 right = cam.right;
        glm::vec3 up    = cam.up;

        cam.lookAt -= right * float(dx) * panSpeed;
        cam.lookAt += up    * float(dy) * panSpeed;

        camchanged = true;
    }

end:
    lastX = xpos;
    lastY = ypos;
}


void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (MouseOverImGuiWindow()) return;

    // Sensitivity (tune this)
    const float zoomSpeed = 0.1f;

    // Trackpad-safe (continuous)
    zoom *= std::exp(-yoffset * zoomSpeed);

    // Clamp zoom
    zoom = glm::clamp(zoom, 0.1f, 1000.0f);

    camchanged = true;
}
