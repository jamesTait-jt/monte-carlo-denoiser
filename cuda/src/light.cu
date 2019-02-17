#include "light.h"
#include "util.h"
#include "triangle.h"
#include "sphere.h"
#include "ray.cuh"

#include <iostream>


Light::Light() {

}

Light::Light(float intensity, vec3 colour, vec4 position) {
    this->intensity_ = intensity;
    this->colour_ = colour;
    this->position_ = position;
}

__device__
vec3 Light::directLight(
    const Intersection & intersection,
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres
) {

    // Distance from point to light source
    float dist_point_to_light = glm::distance(intersection.position, this->position_);

    // normal pointing out from the surface
    vec3 surface_normal = vec3(intersection.normal);

    // direction from surface point to light source
    vec3 surface_to_light_dir = vec3(this->position_ - intersection.position);
    surface_to_light_dir = glm::normalize(surface_to_light_dir);

    // 0.001f is added to the position towards the light to avoid floating point errors
    Ray surface_to_light_ray(
        intersection.position + 0.001f * vec4(surface_to_light_dir, 1.0f),
        vec4(surface_to_light_dir, 1.0f)
    );

    if (surface_to_light_ray.closestIntersection(triangles, num_tris, spheres, num_spheres)) {
        float dist_point_to_intersection = glm::distance(
            intersection.position, 
            surface_to_light_ray.closest_intersection_.position
        ); 
        
        if (dist_point_to_intersection < dist_point_to_light) {
            return vec3(0.0f);
        }
    }

    float max_dot = max(glm::dot(surface_to_light_dir, surface_normal), 0.0f);
    float divisor = 4.0f * (float)M_PI * std::pow(dist_point_to_light, 2.0f);
    float scalar = max_dot / divisor;

    vec3 amount = this->intensity_ * this->colour_;
    vec3 scaled_amount = amount * scalar;
    
    return scaled_amount;// * shapes[intersection.index]->get_material().get_diffuse_light_component();
}
