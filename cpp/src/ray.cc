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
    int num_shapes = shapes.size();
    for (int i = 0 ; i < num_shapes ; i++) {
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
