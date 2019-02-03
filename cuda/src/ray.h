#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>

#include "triangle.h"

#include <vector>

using glm::vec4;
using glm::vec3;

class Triangle;

struct Intersection {
    vec4 position;
    float distance;
    vec4 normal;
    int index;
};

// This class represents a ray of light. It consists of a start point and a
// direction so it can be thought of as a vector.
class Ray {

    public:
        vec4 start_;
        vec4 direction_;
        Intersection closest_intersection_;

        Ray(vec4 start, vec4 direction);
        bool closestIntersection(Triangle * triangles, int num_shapes);
        void rotateRay(float yaw);
        
        vec4 get_start();
        vec4 get_direction();
        Intersection get_closest_intersection();

        void set_start(vec4 start);
        void set_closest_intersection(Intersection intersection);
    
};

#endif
