// This is the parent class for all renderable shapes. Examples of child classes
// are triangle and sphere.
#ifndef SHAPE_H
#define SHAPE_H

#include <glm/glm.hpp>

#include "material.h"
#include "ray.h"

// Creates a shape object. 
class Shape {

    public:
        Material material_;

        Shape(Material material);
        
        virtual bool intersects(Ray * ray, int shape_index)=0;
};

#endif
