#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <glm/glm.hpp>

#include "material.h"
#include "shape.h"

using glm::vec4;

// The Triangle class makes up the majority of rendered shapes. All polygons bar
// spheres can be accurately and efficiently represented with triangles.
class Triangle : public Shape {

    public:
        Triangle(vec4 v0, vec4 v1, vec4 v2, Material material);
        
        bool intersects(Ray * ray, int triangle_index);
        
        vec4 getV0();
        vec4 getV1();
        vec4 getV2();
        vec4 getNormal();

    private:
	vec4 v0_;
	vec4 v1_;
	vec4 v2_;
        vec4 normal_;

        vec4 computeNormal();
};

#endif
