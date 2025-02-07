#ifndef RAY_H
#define RAY_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#include <curand.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>

#include <vector>

using glm::vec4;
using glm::vec3;

class Triangle;
class Sphere;
class Light;
class LightSphere;

struct Intersection {
    vec4 position;
    float distance;
    vec4 normal;
    int index;
    bool is_triangle;
};

// This class represents a ray of light. It consists of a start point and a
// direction so it can be thought of as a vector.
class Ray {

    public:
        vec4 start_;
        vec4 direction_;
        Intersection closest_intersection_;

        CUDA_DEV Ray(
            vec4 start,
            vec4 direction
        );

        CUDA_DEV vec3 raytrace(
            Triangle * triangles,
            int num_tris,
            Sphere * spheres,
            int num_spheres,
            Light light,
            LightSphere
        );

        CUDA_DEV bool closestIntersection(
            Triangle * triangles,
            int num_tris,
            Sphere * spheres,
            int num_spheres
        );

        CUDA_DEV bool intersects(
            Triangle tri,
            int triangle_index
        );

        CUDA_DEV bool intersects(
            Sphere sphere,
            int sphere_index
        );

        CUDA_DEV vec3 tracePath(
            Triangle * triangles,
            int num_tris,
            Sphere * spheres,
            int num_spheres,
            curandState & local_rand_state,
            int num_bounces,
            int curr_depth,
            vec3 & first_sn,
            vec3 & first_albedo,
            float & first_depth
        );

        CUDA_DEV void rotateRay(
            float yaw
        );
};

CUDA_DEV vec3 tracePathIterative(
    Ray ray,
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    curandState * rand_state,
    int num_bounces,
    vec3 & first_sn,
    vec3 & first_albedo,
    float & first_depth
);

#endif
