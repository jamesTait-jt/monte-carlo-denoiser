#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <glm/glm.hpp>

#include "material.h"
#include "ray.cuh"

using glm::vec4;
using glm::mat3;

// The Triangle class makes up the majority of rendered shapes. All polygons bar
// spheres can be accurately and efficiently represented with triangles.
class Triangle {

    public:
	vec4 v0_;
	vec4 v1_;
	vec4 v2_;
        vec4 normal_;
        Material material_;

        Triangle(vec4 v0, vec4 v1, vec4 v2, Material material);
        
        void computeAndSetNormal();
        //bool intersects(Ray * ray, int triangle_index);
        bool cramer(mat3 A, vec3 b, vec3 & solution);
        
        vec4 get_v0();
        vec4 get_v1();
        vec4 get_v2();
        vec4 get_normal();

        void set_v0(vec4 v0);
        void set_v1(vec4 v1);
        void set_v2(vec4 v2);

    private:
        vec4 computeNormal();
};

#endif
