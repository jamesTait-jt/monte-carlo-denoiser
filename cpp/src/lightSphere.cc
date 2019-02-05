#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "lightSphere.h" 

#include <iostream>

LightSphere::LightSphere(vec4 centre, float radius, int num_lights, float intensity, vec3 colour) { 
    this->centre_ = centre;
    this->radius_ = radius;
    this->intensity_ = intensity;
    this->colour_ = colour;
    this->point_lights_ = sphereSample(num_lights, colour);
}

vec3 LightSphere::directLight(Intersection intersection, std::vector<Shape *> shapes) {
    vec3 colour(0,0,0);
    int size = point_lights_.size();
    for (int i = 0 ; i < point_lights_.size() ; i++) {
        Light point_light = point_lights_[i];
        vec3 direct_light = point_light.directLight(intersection, shapes);
        colour = colour + direct_light;
    }

    vec3 final_colour = colour / (float)size;

    return final_colour;
}

std::vector<Light> LightSphere::sphereSample(int num_lights, vec3 colour) {
    std::vector<Light> samples;
    for (int i = 0 ; i < num_lights ; i++) {
        bool contained = true;
        // rejection sampling
        while (contained) {
            float randx = ((float) rand() / (RAND_MAX)) * radius_ - radius_ / 2;
            float randy = ((float) rand() / (RAND_MAX)) * radius_ - radius_ / 2;
            float randz = ((float) rand() / (RAND_MAX)) * radius_ - radius_ / 2;
            vec4 random_point(centre_.x + randx, centre_.y + randy, centre_.z + randz, 1);
            if (containedInSphere(random_point)) {
                Light light(intensity_, colour, random_point);
                samples.push_back(light);
                contained = false;
            }
        }
    }
    return samples;
}

bool LightSphere::containedInSphere(vec4 p) {
    return glm::distance(p, centre_) <= radius_;
}

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
