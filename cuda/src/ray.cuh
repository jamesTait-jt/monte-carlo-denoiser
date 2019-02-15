#ifndef RAY_H
#define RAY_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#include <glm/glm.hpp>

#include <vector>

using glm::vec4;
using glm::vec3;

class Triangle;

struct Intersection {
    vec4 position;
    float distance;
    vec4 normal;
    int index;
};

// This class represents a ray of light. It consists of a start point and a
// direction so it can be thought of as a vector.
class Ray {

    public:
        vec4 start_;
        vec4 direction_;
        Intersection closest_intersection_;

        CUDA_DEV Ray(vec4 start, vec4 direction);
        CUDA_DEV bool closestIntersection(Triangle * triangles, int num_shapes);
        CUDA_DEV void rotateRay(float yaw);
};

#endif
