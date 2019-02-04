#include <iostream>

#include "constants/screen.h"
#include "ray.h"

using glm::mat4;

Ray::Ray(vec4 start, vec4 direction) {
    start_ = start;
    vec3 direction3(direction);
    vec3 normalised_direction3(glm::normalize(direction3));
    direction_ = vec4(normalised_direction3, 1);
    closest_intersection_.distance = std::numeric_limits<float>::max();
}

__device__
void cramer_(mat3 A, vec3 b, vec3 & solution, bool & det_not_zero) {
    // Initialise the solution output
    solution = vec3(0, 0, 0);
    float detA = glm::determinant(A);
    if (detA != 0) {
        det_not_zero = true;
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
}

__device__
void intersects_(Triangle tri, Ray ray, int triangle_index, bool * has_intersection_ptr, Intersection * closest_intersection_ptr) {
    vec4 start = ray.start_;
    vec4 dir = ray.direction_;

    dir = vec4(vec3(dir) * (float)screen_height, 1);

    vec4 v0 = tri.v0_;
    vec4 v1 = tri.v1_;
    vec4 v2 = tri.v2_;

    vec3 v1_minus_v0 = vec3(v1 - v0);
    vec3 v2_minus_v0 = vec3(v2 - v0);
    vec3 start_minus_v0 = vec3(start - v0);

    mat3 A(vec3(-dir), v1_minus_v0, v2_minus_v0);

    vec3 solution;
    bool det_not_zero = false;
    bool * det_not_zero_ptr = &det_not_zero;
    cramer_(A, start_minus_v0, solution, *det_not_zero_ptr);

    if (det_not_zero && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
        if (solution.x < ray.closest_intersection_.distance) {

            //Intersection intersection;
            //intersection.position = start + solution.x * dir;
            //intersection.distance = solution.x;
            //intersection.index = triangle_index;
            //intersection.normal = tri.normal_;
            //*closest_intersection_ptr = intersection;

            closest_intersection_ptr->position = start + solution.x * dir;
            closest_intersection_ptr->distance = solution.x;
            closest_intersection_ptr->index = triangle_index;
            closest_intersection_ptr->normal = tri.normal_;
            *has_intersection_ptr = true;
        }
    } else {
        //printf("\ndet is zero\n");
    }
}

// Gets the closest intersection point of the ray in the scene. Returns false if
// there is no intersection, otherwise sets the member variable
// "closest_intersection_".
__global__
void closestIntersection_(Triangle * triangles, int num_shapes, bool * has_intersection_ptr, Ray ray, Intersection * closest_intersection_ptr) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index ; i < num_shapes ; i+=stride) {
        intersects_(triangles[i], ray, i, has_intersection_ptr, closest_intersection_ptr);
        //*has_intersection = true;
    }
}

bool Ray::closestIntersection(Triangle * triangles, int num_shapes) {
    bool has_intersection = false;
    bool * has_intersection_ptr = &has_intersection;
    cudaMallocManaged(&has_intersection_ptr, sizeof(bool));

    Intersection closest_intersection;
    Intersection * closest_intersection_ptr = &closest_intersection;
    cudaMallocManaged(&closest_intersection_ptr, sizeof(Intersection));

    closestIntersection_<<<1, 256>>>(triangles, num_shapes, has_intersection_ptr, *this, closest_intersection_ptr);
    
    cudaDeviceSynchronize();

    bool return_val = *has_intersection_ptr;
    closest_intersection_ = *closest_intersection_ptr;

    //std::cout << return_val << std::endl;

    return return_val;
}

void Ray::rotateRay(float yaw) {
    mat4 rotation_matrix = mat4(1.0);
    rotation_matrix[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    rotation_matrix[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    this->direction_ = rotation_matrix * this->direction_;
}

vec4 Ray::get_start() {
    return this->start_;
}

vec4 Ray::get_direction() {
    return this->direction_;
}

Intersection Ray::get_closest_intersection() {
    return this->closest_intersection_;
}

void  Ray::set_start(vec4 start) {
    this->start_ = start;
}

void Ray::set_closest_intersection(Intersection intersection) {
    this->closest_intersection_ = intersection;
}
