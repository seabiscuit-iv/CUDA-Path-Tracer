#include "myoptix.h"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <fmt/format.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"

// only include this once
#include <optix_function_table_definition.h>

#include <nvrtc.h>
#include <filesystem>

constexpr bool OPTIX_DEBUG_MODE = true;

bool optix_initialized = false;
OptixDeviceContext optix;

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << level << "][" << tag << "]: "
              << message << "\n";
}

void init_optix() {
    cudaFree(0);
    try {
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions optx_options;
        optx_options.logCallbackFunction = &context_log_cb;
        optx_options.logCallbackLevel = 4;
        optx_options.validationMode = OPTIX_DEBUG_MODE ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        CUcontext cu_ctx = 0;
        OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &optx_options, &optix ) );
    } 
    catch (const std::exception& e) {
        std::cerr << "OptiX initialization failed: " << e.what() << std::endl;
        exit(1);
    }
    std::cout << "OptiX initialized successfully!" << std::endl;
    optix_initialized = true;
}

OptixDeviceContext get_optix() {
    if (!optix_initialized) {
        std::cerr << "Attempted to call get_optix() before optix was initialized" << std::endl;
        exit(1);
    }
    return optix;
}







// OPTIX CUDA RUNTIME COMPILATION ------------------------------

bool readSourceFile( std::string& str, const std::string& filename )
{
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}


void getCuStringFromFile( std::string& cu, const char* filename )
{
    std::vector<std::string> source_locations;

    const std::string file_dir = (std::filesystem::current_path() / "src" / "optixshaders" / filename).string();

    if( readSourceFile( cu, file_dir ) )
    {
        return;
    }

    throw std::runtime_error( "Couldn't open source file " + std::string( file_dir ) );
}

std::string g_nvrtcLog;

void getInputFromCuString( std::string&                    input,
                                  const char*                     cu_source,
                                  const char*                     name,
                                  const char**                    log_string,
                                  const std::vector<const char*>& compiler_options )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char*> options;

    const std::string base_dir = std::filesystem::current_path().string();

    // // Set sample dir as the primary include path
    // std::string sample_dir;
    // if( sample_directory )   
    // {
    //     sample_dir = std::string( "-I" ) + base_dir + '/' + sample_directory;
    //     options.push_back( sample_dir.c_str() );
    // }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {ABSOLUTE_INCLUDE_DIRS};
    // const char*              rel_dirs[] = {SAMPLES_RELATIVE_INCLUDE_DIRS};

    for( const char* dir : abs_dirs )
    {
        include_dirs.push_back( std::string( "-I" ) + dir );
    }
    // for( const char* dir : rel_dirs )
    // {
    //     include_dirs.push_back( "-I" + base_dir + '/' + dir );
    // }
    for( const std::string& dir : include_dirs)
    {
        options.push_back( dir.c_str() );
    }

    // bool optixir = fileExtensionForLoading() == ".optixir";
    bool optixir = true;
    if( optixir )
        options.push_back( "--optix-ir" );
    
    // options.push_back("--std=c++17");
    // options.push_back("--use_fast_math");

    // Collect NVRTC options
    std::copy( std::begin( compiler_options ), std::end( compiler_options ), std::back_inserter( options ) );

    // JIT compile CU to OPTIXIR/PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve OPTIXIR/PTX code
    size_t input_size = 0;
    if( optixir )
    {
#if CUDA_VERSION >= 12000
        NVRTC_CHECK_ERROR( nvrtcGetOptiXIRSize( prog, &input_size ) );
        input.resize( input_size );
        NVRTC_CHECK_ERROR( nvrtcGetOptiXIR( prog, &input[0] ) );
#else
        throw std::runtime_error( "OptiX IR support for NVRTC is only available with CUDA 12.0+" );
#endif
    }
    else
    {
        NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &input_size ) );
        input.resize( input_size );
        NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &input[0] ) );
    }

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}


void build_optix_accel_structure(
    const std::vector<glm::vec3>& h_verts, 
    const glm::vec3* d_verts, 
    const std::vector<Triangle>& h_triangles, 
    const Triangle* d_triangles,
    OptixTraversableHandle& as_handle,
    CUdeviceptr& d_as_output_buffer
)  {
    OptixDeviceContext optix = get_optix();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    CUdeviceptr d_verts_cuptr = CUdeviceptr(d_verts);

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( h_verts.size() );
    triangle_input.triangleArray.vertexBuffers = &d_verts_cuptr;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
    triangle_input.triangleArray.flags         = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;
    triangle_input.triangleArray.indexFormat   = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.numIndexTriplets = h_triangles.size();
    triangle_input.triangleArray.indexBuffer = CUdeviceptr(d_triangles);
    triangle_input.triangleArray.indexStrideInBytes = sizeof(Triangle);

    OptixAccelBufferSizes as_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        optix,
        &accel_options,
        &triangle_input,
        1,
        &as_buffer_sizes
        ) );
    fmt::println("Accel Structure Buffer Size: {} bytes", as_buffer_sizes.outputSizeInBytes);

    CUdeviceptr d_as_temp_buffer;
    cudaMalloc(
                reinterpret_cast<void**>( &d_as_temp_buffer ),
                as_buffer_sizes.tempSizeInBytes
                );
    cudaMalloc(
                reinterpret_cast<void**>( &d_as_output_buffer ),
                as_buffer_sizes.outputSizeInBytes
                );

    OPTIX_CHECK( optixAccelBuild(
                optix,
                0,                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                  // num build inputs
                d_as_temp_buffer,
                as_buffer_sizes.tempSizeInBytes,
                d_as_output_buffer,
                as_buffer_sizes.outputSizeInBytes,
                &as_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    cudaFree( reinterpret_cast<void*>( d_as_temp_buffer ) );
    cudaDeviceSynchronize();

    fmt::println("Acceleration Structure Construction Complete");
}


void compile_pathtracing_optix_module(OptixModule& module, OptixPipelineCompileOptions& pipeline_compile_options) {
    OptixDeviceContext optix = get_optix();

    OptixModuleCompileOptions module_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues      = 3; // fix later
    pipeline_compile_options.numAttributeValues    = 3; // fix later
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params"; // fix later
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::string shaderfile = "optix_triangle.cu";
    std::string cu, input;

    getCuStringFromFile(cu, shaderfile.c_str());
    getInputFromCuString(input, cu.c_str(), "optix_triangle");

    OPTIX_CHECK_LOG( optixModuleCreate(
        optix,
        &module_compile_options,
        &pipeline_compile_options,
        input.c_str(),
        input.size(),
        LOG, &LOG_SIZE,
        &module
        ) );

    fmt::println("Optix Module Compilation Complete");
}

void create_optix_program_groups(
    const OptixModule& module, 
    OptixProgramGroup& raygen_prog_group, 
    OptixProgramGroup& miss_prog_group, 
    OptixProgramGroup& hitgroup_prog_group
) {
    OptixDeviceContext optix = get_optix();

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &raygen_prog_group
                ) );
    fmt::println("          Created Raygen Program");

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &miss_prog_group
                ) );
    fmt::println("          Created Miss Program");

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &hitgroup_prog_group
                ) );
    fmt::println("          Created Hit Program");

    fmt::println("Optix Program Group Creation Complete");
}