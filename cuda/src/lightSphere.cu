#include "lightSphere.h" 

#include <iostream>

LightSphere::LightSphere(vec4 centre, float radius, int num_lights, float intensity, vec3 colour) { this->centre_ = centre; this->radius_ = radius;
    this->centre_ = centre;
    this->radius_ = radius;
    this->intensity_ = intensity;
    this->colour_ = colour;
    Light * samples;
    cudaMallocManaged(&samples, num_lights * sizeof(Light));
    sphereSample(num_lights, samples);
    this->point_lights_ = samples;
    this->num_point_lights_ = num_lights;
}

__device__
vec3 LightSphere::directLight(Intersection intersection, Triangle * triangles, int num_shapes) {
    vec3 colour(0,0,0);
    for (int i = 0 ; i < num_point_lights_ ; i++) {
        Light point_light = point_lights_[i];
        vec3 direct_light = point_light.directLight(intersection, triangles, num_shapes);
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

/*
std::vector<Light> LightSphere::get_point_lights() {
    return this->point_lights_;
}

vec4 LightSphere::get_centre() {
    return this->centre_;
}

float LightSphere::get_radius() {
    return this->radius_;
}

float LightSphere::get_intensity() {
    return this->intensity_;
}

vec3 LightSphere::get_colour() {
    return this->colour_;
}
*/
