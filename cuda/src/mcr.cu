#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <omp.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <chrono>

#include "constants/config.h"
#include "constants/materials.h"
#include "mcr.h"
#include "triangle.h"
#include "sphere.h"
#include "util.h"

int main (int argc, char* argv[]) {
    
    // Calculate the dimensions of the supersampled image
    const int supersample_width = screen_width * anti_aliasing_factor;
    const int supersample_height = screen_height * anti_aliasing_factor;

    // Pointer to the image on the host (CPU)
    vec3 * host_output = new vec3[supersample_width * supersample_height];

    // Pointer to the image on the device (GPU)
    vec3 * device_output;

    // Pointer to the aliased image on the host (CPU)
    vec3 * host_aliased_output = new vec3[screen_width * screen_height];

    // Pointer to the aliased image on the device (GPU)
    vec3 * device_aliased_output;

    // Allocate memory on CUDA device
    cudaMalloc(&device_output, supersample_width * supersample_height * sizeof(vec3));
    cudaMalloc(&device_aliased_output, screen_width * screen_height * sizeof(vec3));

    // Specify the block and grid dimensions to schedule CUDA threads
    dim3 threads_per_block(8, 8);
    dim3 num_blocks(
        supersample_width / threads_per_block.x,
        supersample_height / threads_per_block.y
    );

    // Create a vector of random states for use on the device
    curandState * device_rand_state;
    cudaMalloc(
        (void **)&device_rand_state,
        supersample_width * supersample_height * sizeof(curandState)
    );

    // Load in the shapes
    int num_tris = 30;
    Triangle * triangles;
    cudaMallocManaged(&triangles, num_tris * sizeof(Triangle));

    int num_spheres = 1;
    Sphere * spheres;
    cudaMallocManaged(&spheres, num_tris * sizeof(Sphere));

    printf("CUDA has been initialised. Begin rendering...\n");
    printf("=============================================\n\n");

    // Load the polygons into the triangles array
    loadShapes(triangles, spheres);

    // Initialise the camera object
    Camera camera(
        cam_start_position,
        cam_start_yaw,
        cam_focal_length
    );

    // Define our area light
    LightSphere light_sphere(
        light_start_position, 
        area_light_radius, 
        num_lights, 
        light_intensity, 
        light_colour
    );

    auto start = std::chrono::high_resolution_clock::now();

    // Launch the CUDA kernel from the host and begin rendering 
    render_init<<<num_blocks, threads_per_block>>>(
        device_rand_state,
        supersample_height,
        supersample_width
    );

    render_kernel<<<num_blocks, threads_per_block>>>(
        device_output,
        supersample_height,
        supersample_width,
        camera,
        light_sphere,
        triangles,
        num_tris,
        spheres,
        num_spheres,
        device_rand_state
    ); 

    // Copy results of rendering back to the host
    cudaMemcpy(
        host_output, 
        device_output, 
        supersample_width * supersample_height * sizeof(vec3), 
        cudaMemcpyDeviceToHost
    ); 

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    int duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    printf("Finished rendering in %dms.\n", duration_in_ms);

    save_image(
        host_output, 
        supersample_height, 
        supersample_width, 
        pre_alias_title
    );

    // Specify different scheduling, this time we assign a thread to each pixel
    // of the output image
    threads_per_block = dim3(8, 8);
    num_blocks = dim3(
        screen_width / threads_per_block.x,
        screen_height / threads_per_block.y
    );

    // Perform anti aliasing
    MSAA<<<num_blocks, threads_per_block>>>(
        device_output,
        device_aliased_output,
        supersample_height,
        supersample_width
    );

    // Copy results of rendering back to the host
    cudaMemcpy(
        host_aliased_output, 
        device_aliased_output, 
        screen_width * screen_height * sizeof(vec3), 
        cudaMemcpyDeviceToHost
    ); 

    // Free CUDA memory
    cudaFree(device_output); 
    cudaFree(device_aliased_output); 
    
    printf("Finished aliasing!\n");

    // Save the aliased image
    save_image(
        host_aliased_output, 
        screen_height, 
        screen_width, 
        aliased_title
    );

    // Clear memory for host
    delete[] host_output;
    delete[] host_aliased_output;

    return 0;
}

// Initialises the random states for each thread with the same seed
__global__ 
void render_init(
    curandState * rand_state,
    int supersample_width,
    int supersample_height
) {
    // Assign a thread to each pixel (x, y)
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Calculate the pixel index in the linearised array
    unsigned int pixel_index = (supersample_height - y - 1) * supersample_width + x;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// Bulk of the rendering is controlled here
__global__
void render_kernel(
    vec3 * output,
    int supersample_width,
    int supersample_height,
    Camera camera,
    LightSphere light_sphere,
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    curandState * rand_state
) {
    // Assign a cuda thread to each pixel (x,y)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // The index of the pixel we are working on when the 2x2 array is linearised
    unsigned int pixel_index = (supersample_height - y - 1) * supersample_width + x;

    // Get the rand state for this thread
    curandState local_rand_state = rand_state[pixel_index];
    
    // Flip the y coordinate
    y = supersample_height - y;

    // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
    vec4 dir(
        (float)x - supersample_width / 2 , 
        (float)y - supersample_height / 2 , 
        camera.focal_length_,
        1
    ); 

    // Create a ray for the given pixel
    Ray ray(camera.position_, dir);
    ray.rotateRay(camera.yaw_);

    // If the ray intersects with an object in the scene, perform monte carlo to
    // obtain a lighting estimate
    if (ray.closestIntersection(triangles, num_tris, spheres, num_spheres)) {

        vec3 colour = tracePath(
            ray.closest_intersection_,
            triangles,
            num_tris,
            spheres,
            num_spheres,
            local_rand_state,
            monte_carlo_max_depth,
            0
        );

        /*
        vec3 colour = monteCarlo(
            ray.closest_intersection_,
            triangles,
            num_tris,
            spheres,
            num_spheres,
            light_sphere,
            local_rand_state,
            monte_carlo_max_depth,
            0
        );
        */

        output[pixel_index] = colour;
    } 
    // if there is no intersection, we set the colour to be black
    else {
        output[pixel_index] = vec3(0.0f);
    }
}

__device__
vec3 tracePath(
    Intersection closest_intersection,
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    curandState rand_state,
    int max_depth,
    int depth
) {
    if (depth >= max_depth) {
        return vec3(0.0f);
    } else {
        vec3 base_colour;
        // We have hit a triangle (not a light source)
        if (closest_intersection.is_triangle) {
            base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;

            vec3 intersection_normal_3 = vec3(closest_intersection.normal);
            vec3 N_t, N_b;
            createCoordinateSystem(intersection_normal_3, N_t, N_b);

            vec3 indirect_estimate = vec3(0);
            float pdf = 1 / (2 * (float)M_PI);
            for (int i = 0 ; i < monte_carlo_num_samples ; i++) {
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

                // Generate our ray from the random direction calculated previously
                Ray random_ray(
                    closest_intersection.position + sample_world * 0.0001f,
                    sample_world
                );

                if (random_ray.closestIntersection(triangles, num_tris, spheres, num_spheres)) {
                    indirect_estimate += r1 * tracePath(
                        random_ray.closest_intersection_,
                        triangles,
                        num_tris,
                        spheres,
                        num_spheres,
                        rand_state,
                        max_depth,
                        depth + 1
                    );
                }
            }
            indirect_estimate /= monte_carlo_num_samples * pdf;
            indirect_estimate *= base_colour;
            return indirect_estimate;
        }
        // We have hit a light source
        else {
            return vec3(4.0f);
            //base_colour = spheres[closest_intersection.index].material_.diffuse_light_component_;
        }

    }
}

// Calculates the indirect and direct light estimation for diffuse objects
__device__
vec3 monteCarlo(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    LightSphere light_sphere,
    curandState rand_state,
    int max_depth,
    int depth
) {
    // If we have exceeded our limit of recursion, return the direct light at
    // this point multiplied by the object's colour
    if (depth >= max_depth) {
        vec3 direct_light = light_sphere.directLight(
            closest_intersection,
            triangles,
            num_tris,
            spheres,
            num_spheres
        );

        vec3 base_colour;
        if (closest_intersection.is_triangle) {
            base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;
        } else {
            base_colour = spheres[closest_intersection.index].material_.diffuse_light_component_;
        }
        return direct_light * base_colour;
    } 
    // Otherwise, we must obtain an indirect lighting estimate for this point
    else {
        vec3 base_colour;
        if (closest_intersection.is_triangle) {
            base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;
        } else {
            base_colour = spheres[closest_intersection.index].material_.diffuse_light_component_;
        }
        vec3 direct_light = light_sphere.directLight(
            closest_intersection,
            triangles,
            num_tris,
            spheres,
            num_spheres
        );
        vec3 indirect_estimate = indirectLight(
            closest_intersection,
            triangles,
            num_tris,
            spheres,
            num_spheres,
            light_sphere,
            rand_state,
            max_depth,
            depth + 1
        );
        return (direct_light + indirect_estimate) * base_colour;
    }
}

__device__
vec3 indirectLight(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    LightSphere light_sphere,
    curandState rand_state,
    int max_depth,
    int depth
) {
    vec3 intersection_normal_3 = vec3(closest_intersection.normal);
    
    vec3 N_t, N_b;
    createCoordinateSystem(intersection_normal_3, N_t, N_b);

    vec3 indirect_estimate = vec3(0);
    float pdf = 1 / (2 * (float)M_PI);
    for (int i = 0 ; i < monte_carlo_num_samples ; i++) {
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

        // Generate our ray from the random direction calculated previously
        Ray random_ray(
            closest_intersection.position + sample_world * 0.0001f,
            sample_world
        );

        if (random_ray.closestIntersection(triangles, num_tris, spheres, num_spheres)) {
               indirect_estimate += r1 * monteCarlo(
               random_ray.closest_intersection_,
               triangles,
               num_tris,
               spheres,
               num_spheres,
               light_sphere,
               rand_state,
               max_depth,
               depth + 1
            );
        }
    } 
    indirect_estimate /= monte_carlo_num_samples * pdf;
    return indirect_estimate;
}

// Given two random numbers between 0 and 1, return a direction to a point on a
// hemisphere
__device__
vec3 uniformSampleHemisphere(const float & r1, const float & r2) {
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sin_theta = sqrtf(1 - r1 * r1);
    float phi = 2 * (float)M_PI * r2;
    float x = sin_theta * cosf(phi);
    float z = sin_theta * sinf(phi);
    return vec3(x, r1, z);
} 

// This function creates a new coordinate system in which the up vector is
// oriented along the shaded point normal
__device__
void createCoordinateSystem(const vec3 & N, vec3 & N_t, vec3 & N_b) {
    if (std::fabs(N.x) > std::fabs(N.y)) {
        N_t = vec3(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
    } else {
        N_t = vec3(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
    }
    N_b = glm::cross(N, N_t);
} 

__global__
void MSAA(
    vec3 * supersampled_image,
    vec3 * aliased_output,
    int supersample_height,
    int supersample_width
) {
    // Assign a cuda thread to each pixel (x,y)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // The index of the pixel we are working on when the 2x2 array is linearised
    unsigned int output_pixel_index = (screen_height - y - 1) * screen_width + x;

    // Multiply x and y by the factor so that every pixel is included
    y *= anti_aliasing_factor;
    x *= anti_aliasing_factor;

    // Average the pixel values in a (factor^2 by factor^2) grid
    vec3 avg_pixel_value(0.0f);
    for (int i = 0 ; i < anti_aliasing_factor ; i++) {
        for (int j = 0 ; j < anti_aliasing_factor ; j++) {
            unsigned int input_pixel_index = (supersample_height - (y + i) - 1) * supersample_width + (x + j);
            avg_pixel_value += supersampled_image[input_pixel_index] / (float)(anti_aliasing_factor * anti_aliasing_factor);
        }
    }
    aliased_output[output_pixel_index] = avg_pixel_value;
}

void update(Camera & camera, Light & light) {
    static int t = SDL_GetTicks();
    /* Compute frame time */
    int t2 = SDL_GetTicks();
    float dt = float(t2-t);
    t = t2;

    std::cout << "Render time: " << dt << "ms." << std::endl;

    /* Update variables*/

    const Uint8* keystate = SDL_GetKeyboardState(NULL);

    if (keystate[SDL_SCANCODE_UP]) {
        camera.moveForwards(0.1);
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
        camera.moveBackwards(0.1);
    }
    if (keystate[SDL_SCANCODE_LEFT]) {
        camera.rotateLeft(0.1);
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
        camera.rotateRight(0.1);
    }
    /*if (keystate[SDL_SCANCODE_A]) {
    }
    if (keystate[SDL_SCANCODE_D]) {
        light.translateRight(0.1);
    }
    if (keystate[SDL_SCANCODE_Q]) {
        light.translateUp(0.1);
    }
    if (keystate[SDL_SCANCODE_E]) {
        light.translateDown(0.1);
    }
    if (keystate[SDL_SCANCODE_W]) {
        light.translateForwards(0.1);
    }
    if (keystate[SDL_SCANCODE_S]) {
        light.translateBackwards(0.1);
    }*/
}

void loadShapes(Triangle * triangles, Sphere * spheres) {
    float cornell_length = 555;			// Length of Cornell Box side.

    vec4 A(cornell_length, 0, 0             , 1);
    vec4 B(0             , 0, 0             , 1);
    vec4 C(cornell_length, 0, cornell_length, 1);
    vec4 D(0             , 0, cornell_length, 1);

    vec4 E(cornell_length, cornell_length, 0             , 1);
    vec4 F(0             , cornell_length, 0             , 1);
    vec4 G(cornell_length, cornell_length, cornell_length, 1);
    vec4 H(0             , cornell_length, cornell_length, 1);

    // Counter to track triangles
    int curr_tris = 0;

    // Triangles now take a material as an argument rather than a colour
    // Floor:
    Triangle floor_triangle_1 = Triangle(C, B, A, m_sol_base3);
    //triangles.push_back(floor_triangle_1);
    triangles[curr_tris] = floor_triangle_1;
    curr_tris++;

    Triangle floor_triangle_2 = Triangle(C, D, B, m_sol_base3);
    //triangles.push_back(floor_triangle_2);
    triangles[curr_tris] = floor_triangle_2;
    curr_tris++;

    // Left wall
    Triangle left_wall_1 = Triangle(A, E, C, m_sol_base02);
    //triangles.push_back(left_wall_1);
    triangles[curr_tris] = left_wall_1;
    curr_tris++;

    Triangle left_wall_2 = Triangle(C, E, G, m_sol_base02);
    //triangles.push_back(left_wall_2);
    triangles[curr_tris] = left_wall_2;
    curr_tris++;

    // Right wall
    Triangle right_wall_1 = Triangle(F, B, D, m_sol_base02);
    //triangles.push_back(right_wall_1);
    triangles[curr_tris] = right_wall_1;
    curr_tris++;

    Triangle right_wall_2 = Triangle(H, F, D, m_sol_base02);
    //triangles.push_back(right_wall_2);
    triangles[curr_tris] = right_wall_2;
    curr_tris++;

    // Ceiling
    Triangle ceiling_1 = Triangle(E, F, G, m_sol_base01);
    //triangles.push_back(ceiling_1);
    triangles[curr_tris] = ceiling_1;
    curr_tris++;

    Triangle ceiling_2 = Triangle(F, H, G, m_sol_base01);
    //triangles.push_back(ceiling_2);
    triangles[curr_tris] = ceiling_2;
    curr_tris++;

    // Back wall
    Triangle back_wall_1 = Triangle(G, D, C, m_sol_yellow);
    //triangles.push_back(back_wall_1);
    triangles[curr_tris] = back_wall_1;
    curr_tris++;

    Triangle back_wall_2 = Triangle(G, H, D, m_sol_yellow);
    //triangles.push_back(back_wall_2);
    triangles[curr_tris] = back_wall_2;
    curr_tris++;

    // ---------------------------------------------------------------------------
    // Short block

    A = vec4(240,0,234,1);  //+120 in z -50 in x
    B = vec4( 80,0,185,1);
    C = vec4(190,0,392,1);
    D = vec4( 32,0,345,1);

    E = vec4(240,165,234,1);
    F = vec4( 80,165,185,1);
    G = vec4(190,165,392,1);
    H = vec4( 32,165,345,1);

    // Front
    //triangles.push_back(Triangle(E, B, A, m_sol_red));
    triangles[curr_tris] = Triangle(E, B, A, m_sol_red);
    curr_tris++;
    //triangles.push_back(Triangle(E, F, B, m_sol_red));
    triangles[curr_tris] = Triangle(E, F, B, m_sol_red);
    curr_tris++;

    // Front
    //triangles.push_back(Triangle(F, D, B, m_sol_red));
    triangles[curr_tris] = Triangle(F, D, B, m_sol_red);
    curr_tris++;
    //triangles.push_back(Triangle(F, H, D, m_sol_red));
    triangles[curr_tris] = Triangle(F, H, D, m_sol_red);
    curr_tris++;

    // BACK
    //triangles.push_back(Triangle(H, C, D, m_sol_red));
    triangles[curr_tris] = Triangle(H, C, D, m_sol_red);
    curr_tris++;
    //triangles.push_back(Triangle(H, G, C, m_sol_red));
    triangles[curr_tris] = Triangle(H, G, C, m_sol_red);
    curr_tris++;

    // LEFT
    //triangles.push_back(Triangle(G, E, C, m_sol_red));
    triangles[curr_tris] = Triangle(G, E, C, m_sol_red);
    curr_tris++;
    //triangles.push_back(Triangle(E, A, C, m_sol_red));
    triangles[curr_tris] = Triangle(E, A, C, m_sol_red);
    curr_tris++;

    // TOP
    //triangles.push_back(Triangle(G, F, E, m_sol_red));
    triangles[curr_tris] = Triangle(G, F, E, m_sol_red);
    curr_tris++;
    //triangles.push_back(Triangle(G, H, F, m_sol_red));
    triangles[curr_tris] = Triangle(G, H, F, m_sol_red);
    curr_tris++;

    // ---------------------------------------------------------------------------
    // Tall block

    A = vec4(443,0,247,1);
    B = vec4(285,0,296,1);
    C = vec4(492,0,406,1);
    D = vec4(334,0,456,1);

    E = vec4(443,330,247,1);
    F = vec4(285,330,296,1);
    G = vec4(492,330,406,1);
    H = vec4(334,330,456,1);

    // Front
   
    //triangles.push_back(Triangle(E, B, A, m_sol_blue));
    triangles[curr_tris] = Triangle(E, B, A, m_sol_blue);
    curr_tris++;
    //triangles.push_back(Triangle(E, F, B, m_sol_blue));
    triangles[curr_tris] = Triangle(E, F, B, m_sol_blue);
    curr_tris++;

    // Front
    //triangles.push_back(Triangle(F, D, B, m_sol_blue));
    triangles[curr_tris] = Triangle(F, D, B, m_sol_blue);
    curr_tris++;
    //triangles.push_back(Triangle(F, H, D, m_sol_blue));
    triangles[curr_tris] = Triangle(F, H, D, m_sol_blue);
    curr_tris++;

    // BACK
    //triangles.push_back(Triangle(H, C, D, m_sol_blue));
    triangles[curr_tris] = Triangle(H, C, D, m_sol_blue);
    curr_tris++;
    //triangles.push_back(Triangle(H, G, C, m_sol_blue));
    triangles[curr_tris] = Triangle(H, G, C, m_sol_blue);
    curr_tris++;

    // LEFT
    //triangles.push_back(Triangle(G, E, C, m_sol_blue));
    triangles[curr_tris] = Triangle(G, E, C, m_sol_blue);
    curr_tris++;
    //triangles.push_back(Triangle(E, A, C, m_sol_blue));
    triangles[curr_tris] = Triangle(E, A, C, m_sol_blue);
    curr_tris++;

    // TOP
    //triangles.push_back(Triangle(G, F, E, m_sol_blue));
    triangles[curr_tris] = Triangle(G, F, E, m_sol_blue);
    curr_tris++;
    //triangles.push_back(Triangle(G, H, F, m_sol_blue));
    triangles[curr_tris] = Triangle(G, H, F, m_sol_blue);
    curr_tris++;

    // ---------------------------------------------------------------------------
    // Sphere

    //Sphere for the right wall
    spheres[0] = Sphere(vec4(0, -1, -0.8, 1), 0.3, m_sol_green);

    // ----------------------------------------------
    // Scale to the volume [-1,1]^3

    for (size_t i = 0 ; i < curr_tris ; ++i) {
        triangles[i].v0_ = (triangles[i].v0_ * (2 / cornell_length));
        triangles[i].v1_ = (triangles[i].v1_ * (2 / cornell_length));
        triangles[i].v2_ = (triangles[i].v2_ * (2 / cornell_length));

        triangles[i].v0_ = (triangles[i].v0_ - vec4(1, 1, 1, 1));
        triangles[i].v1_ = (triangles[i].v1_ - vec4(1, 1, 1, 1));
        triangles[i].v2_ = (triangles[i].v2_ - vec4(1, 1, 1, 1));

        vec4 new_v0 = triangles[i].v0_;
        new_v0.x *= -1;
        new_v0.y *= -1;
        new_v0.w = 1.0;
        triangles[i].v0_ = (new_v0);

        vec4 new_v1 = triangles[i].v1_;
        new_v1.x *= -1;
        new_v1.y *= -1;
        new_v1.w = 1.0;
        triangles[i].v1_ = (new_v1);

        vec4 new_v2 = triangles[i].v2_;
        new_v2.x *= -1;
        new_v2.y *= -1;
        new_v2.w = 1.0;
        triangles[i].v2_ = (new_v2);

        triangles[i].computeAndSetNormal();
    }
}
