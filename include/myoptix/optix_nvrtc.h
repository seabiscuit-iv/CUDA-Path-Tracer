#pragma once

#include <string>
#include <vector>

#define CUDA_NVRTC_OPTIONS  \
  "-std=c++17", \
  "-arch", \
  "compute_75", \
  "-lineinfo", \
  "-use_fast_math", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64"

void getCuStringFromFile( std::string& cu, const char* filename );
void getInputFromCuString( std::string&                    input,
                                  const char*                     cu_source,
                                  const char*                     name,
                                  const char**                    log_string = nullptr,
                                  const std::vector<const char*>& compiler_options = {CUDA_NVRTC_OPTIONS});
