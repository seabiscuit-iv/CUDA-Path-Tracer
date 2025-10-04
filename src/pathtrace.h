#pragma once

#include "scene.h"
#include "utilities.h"

#define BLOCK_SIZE_1D 128

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
