#include <iostream>

#include "constants/config.h"
#include "ray.cuh"
#include "triangle.h"
#include "sphere.h"
#include "util.h"

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
bool Ray::intersects(Triangle tri, int triangle_index) {
    bool has_intersection = false;

    vec4 dir = vec4(vec3(direction_) * (float)screen_height, 1.0f);

    vec4 v0 = tri.v0_;
    vec4 v1 = tri.v1_;
    vec4 v2 = tri.v2_;

    vec3 v1_minus_v0 = vec3(v1 - v0);
    vec3 v2_minus_v0 = vec3(v2 - v0);
    vec3 start_minus_v0 = vec3(start_ - v0);

    mat3 A(vec3(-dir), v1_minus_v0, v2_minus_v0);

    vec3 solution;
    bool det_not_zero = cramer_(A, start_minus_v0, solution);

    if (det_not_zero && solution.x >= 0.0f && solution.y >= 0.0f && solution.z >= 0.0f && solution.y + solution.z <= 1.0f) {
        if (solution.x < closest_intersection_.distance) {
            Intersection intersection;
            intersection.position = start_ + solution.x * dir;
            intersection.distance = solution.x;
            intersection.index = triangle_index;
            intersection.normal = tri.normal_;
            closest_intersection_ = intersection;
            has_intersection = true;
        }
    }
    return has_intersection;
}

__device__
bool Ray::intersects(Sphere sphere, int sphere_index) {
    bool has_intersection = false;
    vec3 start = vec3(start_);
    vec3 dir = vec3(direction_) * (float)screen_height;

    float t0, t1;

    float r = sphere.radius_;
    vec3 centre = vec3(sphere.centre_);

    vec3 L = start - centre;
    float a = glm::dot(dir, dir);
    float b = 2 * glm::dot(dir, L);
    float c = glm::dot(L, L) - r * r;

    if (sphere.solveQuadratic(a, b, c, t0, t1)) {
        if (t0 > t1) {
            swap(t0, t1);
        }
        if (t0 < 0) {
            t0 = t1; // if t0 is negative, let's use t1 instead
        }
        if (t0 >= 0) {
            if (t0 < closest_intersection_.distance && t0 > 0) {
                closest_intersection_.position = start_ + t0 * vec4(dir, 1);
                closest_intersection_.distance = t0;
                closest_intersection_.index = sphere_index;
                vec3 normal = glm::normalize((start + t0 * dir) - centre);
                closest_intersection_.normal = vec4(normal, 1);
                has_intersection = true;
            }
        }
    }
    return has_intersection;
}

// Gets the closest intersection point of the ray in the scene. Returns false if
// there is no intersection, otherwise sets the member variable
// "closest_intersection_".
__device__
bool Ray::closestIntersection(
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres
) {
    bool has_intersection = false;
    for (int i = 0 ; i < num_tris ; i++) {
        if (intersects(triangles[i], i)) {
            has_intersection = true;
            closest_intersection_.is_triangle = true;
        }
    }
    for (int i = 0 ; i < num_spheres ; i++) {
        if (intersects(spheres[i], i)) {
            has_intersection = true;
            closest_intersection_.is_triangle = false;
        }
    }
    return has_intersection;
}

__device__
vec3 Ray::tracePathIterative(
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    curandState & rand_state,
    int num_bounces,
    vec3 & first_sn,
    vec3 & first_albedo,
    float & first_depth
) {
    vec3 accum_colour = vec3(0.0f);
    vec3 mask = vec3(1.0f);

    for (int i = 0 ; i < num_bounces ; i++) {

        // If the ray didn't hit an object, return black
        if (!closestIntersection(triangles, num_tris, spheres, num_spheres)) {
            //return vec3(0.0f);
            return accum_colour;
        }

        // Get the material of the object we have intersected with
        Material material = closest_intersection_.is_triangle ?
            triangles[closest_intersection_.index].material_ :
            spheres[closest_intersection_.index].material_;

        // Get the light that is emitted from the object. This will be 0 for anything
        // other than the light objects
        vec3 emittance = material.emitted_light_component_;

        // Get the surface normal of the intersected object (in 3d not 4d)
        vec3 intersection_normal_3 = vec3(closest_intersection_.normal);
        vec3 N_t, N_b;

        // Create out new coordinate system about out intersection point
        createCoordinateSystem(intersection_normal_3, N_t, N_b);

        // Generate two rando numbers
        float r1 = curand_uniform(&rand_state);
        float r2 = curand_uniform(&rand_state);

        // Sample a random point on the hemisphere surrounding out intersection point
        vec3 sample = uniformSampleHemisphere(r1, r2);

        // Convert the sample from our coordinate space to world space
        vec4 sample_world(
            sample.x * N_b.x + sample.y * intersection_normal_3.x + sample.z * N_t.x,
            sample.x * N_b.y + sample.y * intersection_normal_3.y + sample.z * N_t.y,
            sample.x * N_b.z + sample.y * intersection_normal_3.z + sample.z * N_t.z,
            0
        );

        // Generate our ray from the random direction calculated previously
        start_ = closest_intersection_.position + sample_world * 0.000001f;
        direction_ = sample_world;

        // Compute the BRDF for this ray (assuming Lambertian reflection)
        float cos_theta = glm::dot(vec3(direction_), intersection_normal_3);

        // This block ensures that the first bounce returns the albedo to the parameter in the
        // function arguments, if it is not the first bounce then use a different variable name
        if (i == 0) {
            first_sn = vec3(closest_intersection_.normal);
            first_albedo = material.diffuse_light_component_;
            first_depth = closest_intersection_.distance;
        }

        vec3 albedo = material.diffuse_light_component_;

        accum_colour += mask * emittance;

        // This is implicitly using the BRDF and dividing by the pdf but they cancel out to make 2
        //mask *= (BRDF * cos_theta / pdf);
        mask *= (albedo * 2.0f * cos_theta);

    }
    return accum_colour;
}

__device__
vec3 Ray::tracePath(
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    curandState & rand_state,
    int num_bounces,
    int curr_depth,
    vec3 & first_sn,
    vec3 & first_albedo,
    float & first_depth
) {
    // We have bounced enough times
    if (curr_depth >= num_bounces) {
        return vec3(0.0f);
    }

    // The ray didn't hit an object
    if (!closestIntersection(triangles, num_tris, spheres, num_spheres)) {
        return vec3(0.0f);
    }

    Material material = closest_intersection_.is_triangle ?
            triangles[closest_intersection_.index].material_ :
            spheres[closest_intersection_.index].material_;
    vec3 emittance = material.emitted_light_component_;

    vec3 intersection_normal_3 = vec3(closest_intersection_.normal);

    first_sn = intersection_normal_3;
    first_albedo = material.diffuse_light_component_;
    first_depth = closest_intersection_.distance;

    vec3 N_t, N_b;
    createCoordinateSystem(intersection_normal_3, N_t, N_b);

    float r1 = curand_uniform(&rand_state); // cos(theta) = N.Light Direction
    float r2 = curand_uniform(&rand_state);

    vec3 sample = uniformSampleHemisphere(r1, r2);

    // Convert the sample from our coordinate space to world space
    vec4 sample_world(
        sample.x * N_b.x + sample.y * intersection_normal_3.x + sample.z * N_t.x,
        sample.x * N_b.y + sample.y * intersection_normal_3.y + sample.z * N_t.y,
        sample.x * N_b.z + sample.y * intersection_normal_3.z + sample.z * N_t.z,
        0
    );

    float pdf = 1 / (2 * (float)M_PI);

    // Generate our ray from the random direction calculated previously
    Ray random_ray(
        closest_intersection_.position + sample_world * 0.000001f,
        sample_world
    );

    // Compute the BRDF for this ray (assuming Lambertian reflection)
    float cos_theta = glm::dot(vec3(random_ray.direction_), intersection_normal_3);
    vec3 albedo = material.diffuse_light_component_;
    vec3 BRDF = albedo / (float) M_PI ;

    // Recursively trace reflected light sources.
    vec3 new_sn;
    vec3 new_albedo;
    float new_depth;
    vec3 incoming = r1 * random_ray.tracePath(
        triangles,
        num_tris,
        spheres,
        num_spheres,
        rand_state,
        num_bounces,
        curr_depth + 1,
        new_sn,
        new_albedo,
        new_depth
    );

    return emittance + (BRDF * incoming / pdf);
}

__device__
void Ray::rotateRay(float yaw) {
    mat4 rotation_matrix = mat4(1.0);
    rotation_matrix[0] = vec4(cosf(yaw), 0, sinf(yaw), 0);
    rotation_matrix[2] = vec4(-sinf(yaw), 0, cosf(yaw), 0);
    this->direction_ = rotation_matrix * this->direction_;
}