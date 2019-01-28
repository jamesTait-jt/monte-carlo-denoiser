// This is the parent class for all renderable shapes. Examples of child classes
// are triangle and sphere.
#ifndef SHAPE_H
#define SHAPE_H

#include <glm/glm.hpp>

#include "material.h"

// Creates a shape object. 
class Shape {

    public:
        Shape(Material material);
        
        virtual bool intersects(Ray * ray, int shape_index)=0;
        
        Material get_material();
        
        //void set_material(Material material);

    private:
        Material material_ = Material();
};

#endif
