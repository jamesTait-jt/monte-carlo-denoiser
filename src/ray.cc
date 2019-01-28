#include "ray.h"
#include "shape.h"

Ray::Ray(vec4 start, vec4 direction) {
    start_ = start;
    vec3 direction3(direction);
    vec3 normalised_direction3(glm::normalize(direction3));
    direction_ = vec4(normalised_direction3, 1);
}

// Gets the closest intersection point of the ray in the scene. Returns false if
// there is no intersection, otherwise sets the member variable
// "closest_intersection_".
bool Ray::closestIntersection(std::vector<Shape *> shapes) {
    Intersection closest_intersection;
    closest_intersection.distance = std::numeric_limits<float>::max();
    bool return_val = false;
    for (int i = 0 ; i < shapes.size() ; i++) {
        if (shapes[i]->intersects(this, i)) {
            return_val = true;
        }
    }
    return return_val;
}

vec4 Ray::get_start() {
    return this->start_;
}

vec4 Ray::get_direction() {
    return this->direction_;
}
