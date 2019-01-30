#include <iostream>

#include "ray.h"
#include "shape.h"

using glm::mat4;

Ray::Ray(vec4 start, vec4 direction) {
    start_ = start;
    vec3 direction3(direction);
    vec3 normalised_direction3(glm::normalize(direction3));
    direction_ = vec4(normalised_direction3, 1);
    closest_intersection_.distance = std::numeric_limits<float>::max();
}

// Gets the closest intersection point of the ray in the scene. Returns false if
// there is no intersection, otherwise sets the member variable
// "closest_intersection_".
bool Ray::closestIntersection(std::vector<Shape *> shapes) {
    bool return_val = false;
    //std::cout << shapes.size() << std::endl;
    for (int i = 0 ; i < shapes.size() ; i++) {
        if (shapes[i]->intersects(this, i)) {
            return_val = true;
        }
    }
    return return_val;
}

void Ray::rotateRay(float yaw) {
    mat4 rotation_matrix = mat4(1.0);
    rotation_matrix[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    rotation_matrix[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    this->direction_ = rotation_matrix * this->direction_;
}

vec4 Ray::get_start() {
    return this->start_;
}

vec4 Ray::get_direction() {
    return this->direction_;
}

Intersection Ray::get_closest_intersection() {
    return this->closest_intersection_;
}

void Ray::set_closest_intersection(Intersection intersection) {
    this->closest_intersection_ = intersection;
}
