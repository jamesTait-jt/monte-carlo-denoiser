#include <iostream>

#include "constants/config.h"
#include "ray.cuh"
#include "triangle.h"

using glm::mat4;

__device__
Ray::Ray(vec4 start, vec4 direction) {
    start_ = start;
    vec3 direction3(direction);
    vec3 normalised_direction3(glm::normalize(direction3));
    direction_ = vec4(normalised_direction3, 1);
    closest_intersection_.distance = 999999.0f;
    //closest_intersection_.distance = std::numeric_limits<float>::max();
}

__device__
bool cramer_(mat3 A, vec3 b, vec3 & solution) {
    bool det_not_zero = false;
    // Initialise the solution output
    solution = vec3(0, 0, 0);
    float detA = glm::determinant(A);
    if (detA != 0) {
        det_not_zero = true;
        // Temp variable to hold the value of A
        mat3 temp = A;

        A[0] = b;
        solution.x = glm::determinant(A) / detA;
        A = temp;

        A[1] = b;
        solution.y = glm::determinant(A) / detA;
        A = temp;

        A[2] = b;
        solution.z = glm::determinant(A) / detA;
        A = temp;
    } 
    return det_not_zero;
}

__device__
bool intersects_(Triangle tri, Ray * ray, int triangle_index) {
    bool has_intersection = false;
    vec4 start = ray->start_;
    vec4 dir = ray->direction_;

    dir = vec4(vec3(dir) * (float)screen_height, 1.0f);

    vec4 v0 = tri.v0_;
    vec4 v1 = tri.v1_;
    vec4 v2 = tri.v2_;

    vec3 v1_minus_v0 = vec3(v1 - v0);
    vec3 v2_minus_v0 = vec3(v2 - v0);
    vec3 start_minus_v0 = vec3(start - v0);

    mat3 A(vec3(-dir), v1_minus_v0, v2_minus_v0);

    vec3 solution;
    bool det_not_zero = cramer_(A, start_minus_v0, solution);

    if (det_not_zero && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
        if (solution.x < ray->closest_intersection_.distance) {

            Intersection intersection;
            intersection.position = start + solution.x * dir;
            intersection.distance = solution.x;
            intersection.index = triangle_index;
            intersection.normal = tri.normal_;
            ray->closest_intersection_ = intersection;
            has_intersection = true;
        }
    }
    return has_intersection;
}

// Gets the closest intersection point of the ray in the scene. Returns false if
// there is no intersection, otherwise sets the member variable
// "closest_intersection_".
__device__
bool Ray::closestIntersection(Triangle * triangles, int num_shapes) {
    bool has_intersection = false;
    for (int i = 0 ; i < num_shapes ; i++) {
        if (intersects_(triangles[i], this, i)) {
            has_intersection = true;
        }
    }
    return has_intersection;
}

__device__
void Ray::rotateRay(float yaw) {
    mat4 rotation_matrix = mat4(1.0);
    rotation_matrix[0] = vec4(cosf(yaw), 0, sinf(yaw), 0);
    rotation_matrix[2] = vec4(-sinf(yaw), 0, cosf(yaw), 0);
    this->direction_ = rotation_matrix * this->direction_;
}