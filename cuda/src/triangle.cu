#include <iostream>

#include "triangle.h"

using glm::vec3;

Triangle::Triangle(){}

Triangle::Triangle(vec4 v0, vec4 v1, vec4 v2, Material material) {
   v0_ = v0; 
   v1_ = v1; 
   v2_ = v2; 
   normal_ = computeNormal();
   material_ = material;
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
