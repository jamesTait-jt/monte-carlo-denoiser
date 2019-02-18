#include <string>

#ifndef UTIL_H
#define UTIL_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <glm/glm.hpp>

using glm::vec3;

float clamp(
    float x
);

int scaleTo255(
    float x
);

void save_image(
    vec3 * image,
    int height,
    int width,
    std::string name
);

float max(
    float a,
    float b
);

CUDA_HOSTDEV void swap(
    float & a,
    float & b
);

#endif
