#include "triangle.h"

using glm::vec3;

Triangle::Triangle(vec4 v0, vec4 v1, vec4 v2, Material material) : Shape(material) {
   v0_ = v0; 
   v1_ = v1; 
   v2_ = v2; 
   normal_ = computeNormal();
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

vec4 Triangle::getV0() {
    return this->v0_;
}

vec4 Triangle::getV1() {
    return this->v1_;
}

vec4 Triangle::getV2() {
    return this->v2_;
}

vec4 Triangle::getNormal() {
    return this->normal_;
}
