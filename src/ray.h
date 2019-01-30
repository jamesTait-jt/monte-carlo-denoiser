#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>

#include <vector>

using glm::vec4;
using glm::vec3;

class Shape;

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
        Ray(vec4 start, vec4 direction);
        bool closestIntersection(std::vector<Shape *> shapes);
        void rotateRay(float yaw);       
        
        vec4 get_start();
        vec4 get_direction();
        Intersection get_closest_intersection();
        void set_closest_intersection(Intersection intersection);
    
    private:
        vec4 start_;
        vec4 direction_;
        Intersection closest_intersection_;
};

#endif
