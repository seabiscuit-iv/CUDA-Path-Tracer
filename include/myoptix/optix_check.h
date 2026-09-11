#pragma once

#include <optix.h>
#include <string>
#include <exception>
#include <stdexcept>
#include <fmt/format.h>

#define OPTIX_CHECK(call)                                                      \
do {                                                                           \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                               \
        fprintf(stderr, "OptiX call (%s) failed with code %d\n", #call, res); \
        exit(1);                                                              \
    }                                                                          \
} while(0)

#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x
#define LINE_STR STRINGIFY( __LINE__ )

#define NVRTC_CHECK_ERROR( func )                                                                                           \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if( code != NVRTC_SUCCESS )                                                                                         \
            throw std::runtime_error( fmt::format("ERROR: {} ({}): {}", __FILE__, LINE_STR, std::string( nvrtcGetErrorString( code ) )) ); \
    } while( 0 )

static inline void optixCheckLog( OptixResult  res,
                           const char*  log,
                           size_t       sizeof_log,
                           size_t       sizeof_log_returned,
                           const char*  call,
                           const char*  file,
                           unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        throw std::runtime_error( fmt::format("Optix call '{}' failed: {}: {} \nLog:\n{} {}\n", call, file, line, log, ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" )) );
    }
}

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call,     \
                                __FILE__, __LINE__ );                          \
    } while( false )
