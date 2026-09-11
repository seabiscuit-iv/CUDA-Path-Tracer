#pragma once

#include <optix.h>
#include <vector>

void create_ias(
    const std::vector<OptixInstance>& instances,
    CUdeviceptr &d_optix_instances,
    OptixTraversableHandle& ias_handle
);
