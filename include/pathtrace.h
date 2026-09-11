#pragma once

#include "scene.h"
#include "utilities.h"
#include "config.h"

#include "pathtrace/pathtracer_options.h"
#include "pathtrace/pathtrace_state.h"
#include "pathtrace/pathtrace_buffers.h"

#define BLOCK_SIZE_1D 128

void pathtrace(uchar4* pbo, int frame, int iteration);
