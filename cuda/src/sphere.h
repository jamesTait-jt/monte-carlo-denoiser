#ifndef SPHERE_H
#define SPHERE_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#include <glm/glm.hpp>

#include "material.h"

using glm::vec4;

class Sphere {

    public:
        vec4 centre_;
        float radius_;
        Material material_;

        Sphere(vec4 centre, float radius, Material material);


        CUDA_DEV bool solveQuadratic(
            const float & a,
            const float & b,
            const float & c,
            float & x0,
            float & x1
        );
};

#endif