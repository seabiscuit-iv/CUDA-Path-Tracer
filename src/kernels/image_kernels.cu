#include "kernels/image_kernels.h"

#include "pathtrace/dev_options.h"
#include "tonemapping.h"

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, float iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        float invIter = __frcp_rn(iter);

        pix = pix * invIter;

        if (DEV_OPTIONS.material_debug_mode == 0) {
            if (DEV_OPTIONS.color_mode == 0) {
                pix = pix / (pix + glm::vec3(1.0f));
            }
            else if (DEV_OPTIONS.color_mode == 1) {
                pix = AgX(pix);
            }
            else {
                pix = ACESFilm(pix);
            }

            //gamma correction
            pix = glm::pow(pix, glm::vec3(0.45f));
        }

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}


__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* __restrict__ iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        glm::vec3 color = iterationPath.color;

        image[iterationPath.pixelIndex] += color;
    }
}
