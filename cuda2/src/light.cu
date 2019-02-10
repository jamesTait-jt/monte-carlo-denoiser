#include "light.h"

#include <iostream>

#define max(a,b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

Light::Light() {

}

Light::Light(float intensity, vec3 colour, vec4 position) {
    this->intensity_ = intensity;
    this->colour_ = colour;
    this->position_ = position;
}

__device__
vec3 Light::directLight(const Intersection & intersection, Triangle * triangles, int num_shapes) {

    // Distance from point to light source
    float dist_point_to_light = glm::distance(intersection.position, this->position_);

    // normal pointing out from the surface
    vec3 surface_normal = vec3(intersection.normal);

    // direction from surface point to light source
    vec3 surface_to_light_dir = vec3(this->position_ - intersection.position);
    surface_to_light_dir = glm::normalize(surface_to_light_dir);

    // 0.001f is added to the position towards the light to avoid floating point errors
    Ray surface_to_light_ray(
        intersection.position + 0.001f * vec4(surface_to_light_dir, 1),
        vec4(surface_to_light_dir, 1)
    );

    if (surface_to_light_ray.closestIntersection(triangles, num_shapes)) {
        float dist_point_to_intersection = glm::distance(
            intersection.position, 
            surface_to_light_ray.closest_intersection_.position
        ); 
        
        if (dist_point_to_intersection < dist_point_to_light) {
            return vec3(0);        
        }
    }

    float scalar = (
        max(
            dot(surface_to_light_dir, surface_normal), 
            0.0f
        ) / (4.0f * M_PI * std::pow(dist_point_to_light, 2))
    );
    
    vec3 amount = this->intensity_ * this->colour_;
    vec3 scaled_amount = amount * scalar;
    
    return scaled_amount;// * shapes[intersection.index]->get_material().get_diffuse_light_component();
}

vec4 Light::get_position() {
    return this->position_;
}
