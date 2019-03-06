#include <string>
#include <vector>

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
    std::string title
);

void save_image(
    float * image,
    int height,
    int width,
    std::string title
);

void save_patches(
    vec3 * image,
    int patch_size,
    std::string title,
    std::vector<int> seed
);

float max(
    float a,
    float b
);

CUDA_HOSTDEV void swap(
    float & a,
    float & b
);

__device__
void createCoordinateSystem(
        const vec3 & N,
        vec3 & N_t,
        vec3 & N_b
);

__device__
vec3 uniformSampleHemisphere(
        const float & r1,
        const float & r2
);

__device__
vec3 mean(
    vec3 * data,
    int data_size
);

__device__
float mean(
    float * data,
    int data_size
);

__device__
float mySum(
    vec3 v
);

__device__
float luminance(
    vec3 v
);

#endif
