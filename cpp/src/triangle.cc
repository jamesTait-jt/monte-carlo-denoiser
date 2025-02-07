#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"

#include <iostream>

#include "triangle.h"
#include "constants/screen.h"

using glm::vec3;

Triangle::Triangle(vec4 v0, vec4 v1, vec4 v2, Material material) : Shape(material) {
    v0_ = v0; 
    v1_ = v1; 
    v2_ = v2; 

    e01_ = vec3(v1 - v0);
    e02_ = vec3(v2 - v0);

    normal_ = computeNormal();
}

bool Triangle::intersects(Ray * ray, int triangle_index) {
    bool return_val = false;
    vec4 start = ray->start_;
    vec4 dir = ray->direction_;

    dir = vec4(vec3(dir) * (float)SCREEN_HEIGHT, 1);

    vec3 v1_minus_v0 = vec3(v1_ - v0_);
    vec3 v2_minus_v0 = vec3(v2_ - v0_);
    vec3 start_minus_v0 = vec3(start - v0_);

    //mat3 A(vec3(-dir), v1_minus_v0, v2_minus_v0);
    mat3 A(vec3(-dir), e01_, e02_);

    vec3 solution;
    bool crmr = cramer(A, start_minus_v0, solution);

    if (crmr && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
        if (solution.x < ray->closest_intersection_.distance) {

            Intersection intersection;
            intersection.position = start + solution.x * dir;
            intersection.distance = solution.x;
            intersection.index = triangle_index;
            intersection.normal = normal_;

            ray->closest_intersection_ = intersection;
            return_val = true;
        }
    }
return return_val;
}

// Calculate the surface normal of the triangle. Calculate the cross product
// between two edges, then renormalise, being careful about using homogeneous
// coordinates.
vec4 Triangle::computeNormal() {
    vec3 v1_minus_v0 = vec3(v1_- v0_); 
    vec3 v2_minus_v0 = vec3(v2_ - v0_); 
    vec3 edge_cross_product = glm::normalize(glm::cross(v2_minus_v0, v1_minus_v0));
    vec4 normal = vec4(
        edge_cross_product.x,
        edge_cross_product.y,
        edge_cross_product.z,
        1
    );
    return normal;
}

void Triangle::computeAndSetNormal() {
    vec4 normal = this->computeNormal();
    this->normal_ = normal;
}

bool Triangle::cramer(mat3 A, vec3 b, vec3 & solution) {
    bool ret = false;
    // Initialise the solution output
    solution = vec3(0, 0, 0);
    float detA = glm::determinant(A);
    if (detA != 0) {
        ret = true;
        // Temp variable to hold the value of A
        mat3 temp = A;

        A[0] = b;
        solution.x = determinant(A) / detA;
        A = temp;

        A[1] = b;
        solution.y = determinant(A) / detA;
        A = temp;

        A[2] = b;
        solution.z = determinant(A) / detA;
        A = temp;
    } 
    return ret;
}
