#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <omp.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

#include "constants/screen.h"
#include "constants/materials.h"
#include "mcr.h"
#include "shape.h"
#include "triangle.h"

int main (int argc, char* argv[]) {
    
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

    curandState * device_rand_state;
    cudaMalloc((void **)&device_rand_state,  supersample_width * supersample_height * sizeof(curandState));

    // Load in the shapes
    int num_tris = 30;
    Triangle * triangles;
    cudaMallocManaged(&triangles, num_tris * sizeof(Triangle));

    printf("CUDA has been initialised. Begin rendering...\n");
    printf("=============================================\n\n");

    // Load the polygons into the triangles array
    loadShapes(triangles);

    // Initialise the camera object
    Camera camera(vec4(0, 0, -3, 1));

    // Define our area light
    int num_lights = 1;
    int light_intensity = 12.0f;
    vec3 light_colour(0.75f, 0.75f, 0.75f);
    LightSphere light_sphere(
        vec4(0.0f, -0.4f, -0.9f, 1.0f), 
        0.1f, 
        num_lights, 
        light_intensity, 
        light_colour
    );

    time_t start = time(0);

    // Launch the CUDA kernel from the host and begin rendering 
    render_init<<<num_blocks, threads_per_block>>>(
        device_rand_state
    );

    render_kernel<<<num_blocks, threads_per_block>>>(
        device_output,
        camera,
        light_sphere,
        triangles,
        num_tris,
        device_rand_state
    ); 

    // Copy results of rendering back to the host
    cudaMemcpy(
        host_output, 
        device_output, 
        supersample_width * supersample_height * sizeof(vec3), 
        cudaMemcpyDeviceToHost
    ); 

    time_t end = time(0);
    double time = difftime(end, start);
    printf("Finished rendering in %fs.\n", time);

    const char * pre_alias_title = "pre_alias.ppm";
    save_image(host_output, supersample_height, supersample_width, pre_alias_title);

    // Specify the block and grid dimensions to schedule CUDA threads
    threads_per_block = dim3(8, 8);
    num_blocks = dim3(
        screen_width / threads_per_block.x,
        screen_height / threads_per_block.y
    );

    MSAA<<<num_blocks, threads_per_block>>>(
        device_output,
        device_aliased_output
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

    const char * aliased_name = "aliased.ppm"; 
    save_image(host_aliased_output, screen_height, screen_width, aliased_name);

    // Clear memory for host output
    delete[] host_output;
    delete[] host_aliased_output;

    return 0;
}

float clamp(float x) { 
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; 
} 

// convert RGB float in range [0,1] to int in range [0, 255]
int scaleTo255(float x) {
    return int(pow(clamp(x), 1) * 255 + .5); 
}

void save_image(vec3 * image, int height, int width, const char * name) {
    FILE * file = fopen(name, "w");
    fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
    for(int i = 0 ; i < width * height; i++) {
        fprintf(file, "%d %d %d ", scaleTo255(image[i].x), 
                                   scaleTo255(image[i].y),
                                   scaleTo255(image[i].z)
        );
    }
    printf("Saved image to '%s'\n", name);
}

__global__ 
void render_init(curandState * rand_state) {
   unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
   unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
   unsigned int pixel_index = (supersample_height - y - 1) * supersample_width + x;

   //Each thread gets same seed, a different sequence number, no offset
   curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__
void render_kernel(
    vec3 * output, 
    Camera camera, 
    LightSphere light_sphere,
    Triangle * triangles,
    int num_tris,
    curandState * rand_state
) {
    // Assign a cuda thread to each pixel (x,y)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("%f %f %f\n", triangles[0].material_.diffuse_light_component_.x);

    // The index of the pixel we are working on when the 2x2 array is linearised
    unsigned int pixel_index = (supersample_height - y - 1) * supersample_width + x;
    curandState local_rand_state = rand_state[pixel_index];
    
    // Flip the y coordinate
    y = supersample_height - y;

    // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
    vec4 dir(((float)x - supersample_width / 2) , ((float)y - supersample_height / 2) , focal_length , 1); 

    // Create a ray for the given pixel
    Ray ray(camera.position_, dir);
    ray.rotateRay(camera.yaw_);

    if (ray.closestIntersection(triangles, num_tris)) {
        Intersection closest_intersection = ray.closest_intersection_;
        //vec3 direct_light = light.directLight(closest_intersection, shapes);
        //vec3 direct_light = light_sphere.directLight(closest_intersection, triangles, num_tris);
        //vec3 base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;
        //printf("%d %d %d\n", base_colour.x, base_colour.y, base_colour.z);
        vec3 colour = monteCarlo(
            closest_intersection,
            triangles,
            num_tris,
            light_sphere,
            local_rand_state,
            monte_carlo_depth,
            0
        );
        //output[pixel_index] = direct_light * base_colour;
        output[pixel_index] = colour;
    } else {
        output[pixel_index] = vec3(0.5);
    }
}

// Calculates the indirect and direct light estimation for diffuse objects
__device__
vec3 monteCarlo(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris,
    LightSphere light_sphere,
    curandState rand_state,
    int max_depth,
    int depth
) {
    if (depth >= max_depth) {
        vec3 direct_light = light_sphere.directLight(
            closest_intersection,
            triangles,
            num_tris
        );
        vec3 base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;
        return direct_light * base_colour;
    }

    vec3 intersection_normal_3 = vec3(closest_intersection.normal);
    vec3 base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;
    vec3 direct_light = light_sphere.directLight(
        closest_intersection,
        triangles,
        num_tris
    );

    vec3 N_t, N_b;
    createCoordinateSystem(intersection_normal_3, N_t, N_b);

    vec3 indirect_estimate = vec3(0);
    float pdf = 1 / (2 * M_PI);
    for (int i = 0 ; i < monte_carlo_samples ; i++) {
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

        Ray random_ray(
            closest_intersection.position + sample_world * 0.0001f,
            sample_world
        );

        if (random_ray.closestIntersection(triangles, num_tris)) {
               indirect_estimate += r1 * monteCarlo(
               random_ray.closest_intersection_,
               triangles,
               num_tris,
               light_sphere,
               rand_state,
               max_depth,
               depth + 1
            );
        } else {
            //i--;
        }
    } 
    indirect_estimate /= (float) (monte_carlo_samples * pdf);
    return (direct_light + indirect_estimate) * base_colour;
}

// Given two random numbers between 0 and 1, return a direction to a point on a
// hemisphere
__device__
vec3 uniformSampleHemisphere(const float & r1, const float & r2) {
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sin_theta = sqrtf(1 - r1 * r1);
    float phi = 2 * M_PI * r2;
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
void MSAA(vec3 * supersampled_image, vec3 * aliased_output) {
    // Assign a cuda thread to each pixel (x,y)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // The index of the pixel we are working on when the 2x2 array is linearised
    unsigned int output_pixel_index = (screen_height - y - 1) * screen_width + x;

    // Calculate how many times bigger the supersampled image is
    int factor = supersample_width / screen_width;

    // Multiply x and y by the factor so that every pixel is included
    y *= factor;
    x *= factor;

    // Average the pixel values in a (factor^2 by factor^2) grid
    vec3 avg_pixel_value(0.0f);
    for (int i = 0 ; i < factor ; i++) {
        for (int j = 0 ; j < factor ; j++) {
            unsigned int input_pixel_index = (supersample_height - (y + i) - 1) * supersample_width + (x + j);
            avg_pixel_value += supersampled_image[input_pixel_index] / (float)(factor * factor);
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

void renderImageBuffer(vec3 * image, SdlWindowHelper sdl_window) {
    for (int x = 0 ; x < screen_height ; x++) {
        for (int y = 0 ; y < screen_width ; y++) {
           sdl_window.putPixel(x, y, image[x * screen_width + y]); 
        }
    }
    sdl_window.render();
}

void loadShapes(Triangle * triangles) {
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

    // ----------------------------------------------
    // Scale to the volume [-1,1]^3

    for (size_t i = 0 ; i < curr_tris ; ++i) {
        triangles[i].set_v0(triangles[i].get_v0() * (2 / cornell_length));
        triangles[i].set_v1(triangles[i].get_v1() * (2 / cornell_length));
        triangles[i].set_v2(triangles[i].get_v2() * (2 / cornell_length));

        triangles[i].set_v0(triangles[i].get_v0() - vec4(1, 1, 1, 1));
        triangles[i].set_v1(triangles[i].get_v1() - vec4(1, 1, 1, 1));
        triangles[i].set_v2(triangles[i].get_v2() - vec4(1, 1, 1, 1));

        vec4 new_v0 = triangles[i].get_v0();
        new_v0.x *= -1;
        new_v0.y *= -1;
        new_v0.w = 1.0;
        triangles[i].set_v0(new_v0);

        vec4 new_v1 = triangles[i].get_v1();
        new_v1.x *= -1;
        new_v1.y *= -1;
        new_v1.w = 1.0;
        triangles[i].set_v1(new_v1);

        vec4 new_v2 = triangles[i].get_v2();
        new_v2.x *= -1;
        new_v2.y *= -1;
        new_v2.w = 1.0;
        triangles[i].set_v2(new_v2);

        triangles[i].computeAndSetNormal();
    }
}
