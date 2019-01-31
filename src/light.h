#ifndef LIGHT_H
#define LIGHT_H

#include <glm/glm.hpp>

#include <vector>

#include "shape.h"
#include "ray.h"

using glm::vec3;

// This class is a simple point light implementation that can be moved around
// the room.
class Light {
    
    public:
        Light(float intensity, vec3 colour, vec4 position);
        vec3 directLight(const Intersection & intersection, std::vector<Shape *> shapes);

        float get_intensity();
        vec3 get_colour();
        vec4 get_position();

    private:
        float intensity_;
        vec3 colour_;
        vec4 position_;

};

#endif
