#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "pathtrace/pathtracer_options.h"

extern __constant__ PathTracerOptions DEV_OPTIONS;
