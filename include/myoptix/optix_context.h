#pragma once

#include <optix.h>
#include <optix_stubs.h>

extern bool optix_initialized;
extern OptixDeviceContext optix;

void init_optix();
OptixDeviceContext get_optix();
