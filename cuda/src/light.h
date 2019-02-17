#ifndef LIGHT_H
#define LIGHT_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#include <glm/glm.hpp>

#include <vector>

class Triangle;
class Sphere;
struct Intersection;

using glm::vec3;
using glm::vec4;

// This class is a simple point light implementation that can be moved around
// the room.
class Light {
    
    public:
        float intensity_;
        vec3 colour_;
        vec4 position_;

        Light();
        Light(
            float intensity,
            vec3 colour,
            vec4 position
        );

        CUDA_DEV vec3 directLight(
            const Intersection & intersection,
            Triangle * triangles,
            int num_shapes,
            Sphere * spheres,
            int num_spheres
        );
};

#endif
