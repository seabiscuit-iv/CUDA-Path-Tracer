#pragma once

#include <optix.h>
#include <optix_stubs.h>

void init_optix();
OptixDeviceContext get_optix();


#define OPTIX_CHECK(call)                                                      \
do {                                                                           \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                               \
        fprintf(stderr, "OptiX call (%s) failed with code %d\n", #call, res); \
        exit(1);                                                              \
    }                                                                          \
} while(0)