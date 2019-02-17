#include "lightSphere.h" 

#include "triangle.h"
#include "sphere.h"
#include "light.h"
#include "ray.cuh"

#include <iostream>

LightSphere::LightSphere(
    vec4 centre,
    float radius,
    int num_lights,
    float intensity,
    vec3 colour
) {
    centre_ = centre;
    radius_ = radius;
    intensity_ = intensity;
    colour_ = colour;
    Light * samples;
    cudaMallocManaged(&samples, num_lights * sizeof(Light));

    sphereSample(num_lights, samples);
    point_lights_ = samples;
    num_point_lights_ = num_lights;
}

__device__
vec3 LightSphere::directLight(
    Intersection intersection,
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres
) {
    vec3 colour(0,0,0);
    for (int i = 0 ; i < num_point_lights_ ; i++) {
        Light point_light = point_lights_[i];
        vec3 direct_light = point_light.directLight(intersection, triangles, num_tris, spheres, num_spheres);
        colour = colour + direct_light;
    }

    vec3 final_colour = colour / (float)num_point_lights_;

    return final_colour;
}

void LightSphere::sphereSample(int num_lights, Light * samples) {
    for (int i = 0 ; i < num_lights ; i++) {
        bool contained = true;
        // rejection sampling
        while (contained) {
            float randx = ((float) rand() / (RAND_MAX)) * radius_ - radius_ / 2;
            float randy = ((float) rand() / (RAND_MAX)) * radius_ - radius_ / 2;
            float randz = ((float) rand() / (RAND_MAX)) * radius_ - radius_ / 2;
            vec4 random_point(centre_.x + randx, centre_.y + randy, centre_.z + randz, 1);
            if (containedInSphere(random_point)) {
                Light light(intensity_, colour_, random_point);
                samples[i] = light;
                contained = false;
            }
        }
    }
}

bool LightSphere::containedInSphere(vec4 p) {
    return glm::distance(p, centre_) <= radius_;
}
