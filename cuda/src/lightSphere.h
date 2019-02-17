#ifndef LIGHTSPHERE_H
#define LIGHTSPHERE_H

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#include <glm/glm.hpp>

#include <vector>

class Triangle;
class Sphere;
class Light;
struct Intersection;

using glm::vec3;
using glm::vec4;

// This class is a sphere containing multiple point lights. Using this method
// gives us nice soft shadows rather than sharply moving from light to dark
class LightSphere {

    public:
        Light * point_lights_;
        vec4 centre_;
        float radius_;
        float intensity_;
        vec3 colour_;
        int num_point_lights_;

        LightSphere(
            vec4 centre,
            float radius,
            int num_lights,
            float intensity,
            vec3 colour
        );

        CUDA_DEV vec3 directLight(
            Intersection intersection,
            Triangle * triangles,
            int num_tris,
            Sphere * spheres,
            int num_spheres
        );

    private:
        void sphereSample(int numLights, Light * samples);
        bool containedInSphere(vec4 p);

};

#endif
