#include "common.h"

__device__ int cudaUtils::select_from_cdf(float* cdf, int size, float xi) {
    int low = 0;
    int high = size - 1;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (cdf[mid] < xi) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}
