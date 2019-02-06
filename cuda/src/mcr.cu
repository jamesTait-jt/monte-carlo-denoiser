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

    //omp_set_num_threads(4);
    
    // 30 triangles at the moment
    int num_tris = 30;

    // Allocate unified memory for access from CPU or GPU
    //std::vector<Triangle> triangles;
    Triangle * triangles;
    cudaMallocManaged(&triangles, num_tris * sizeof(Triangle));

    loadShapes(triangles);

    int num_lights = 1;
    float light_intensity = 10.0f;
    vec3 light_colour = vec3(0.75);

    Camera camera(vec4(0, 0, -3, 1));
    Light light(10.0f, vec3(1), vec4(0, -0.4, -0.9, 1.0));
    LightSphere light_sphere(
        vec4(0, -0.4, -0.9, 1.0), 
        0.1f, 
        num_lights, 
        light_intensity, 
        light_colour
    );
    
    SdlWindowHelper sdl_window(screen_width, screen_height);
    
    int i = 0;
    while(sdl_window.noQuitMessage() && i < 10000) {
        i++;
        update(camera, light);
        draw(camera, light, light_sphere, triangles, num_tris, sdl_window);
    }

    sdl_window.destroy();

    //cudaFree(shapes);
    cudaFree(triangles);

    std::cout << "boop" << std::endl;
    return 0;
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
        light.translateLeft(0.1);
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

__global__
void draw_(
    Camera camera, 
    Light & light, 
    LightSphere light_sphere, 
    Triangle * triangles, 
    int num_shapes, 
    vec3 * image, 
    int screen_height, 
    int screen_width, 
    int focal_length, 
    int seed,
    int monte_carlo_samples
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index ; i < screen_height * screen_width ; i += stride) {
        int x = i / screen_height;
        //printf("%d\n", x);
        int y = i % screen_height;
//        for (int y = 0 ; y < screen_width ; y++) {
        // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
        vec4 dir((x - screen_width / 2) , (y - screen_height / 2) , focal_length , 1);

        // Create a ray that we will change the direction for below
        Ray ray(camera.position_, dir);
        ray.rotateRay(camera.yaw_);

        if (ray.closestIntersection(triangles, num_shapes)) {
            Intersection closest_intersection = ray.closest_intersection_;
            //vec3 direct_light = light.directLight(closest_intersection, shapes);
            vec3 direct_light = light_sphere.directLight(closest_intersection, triangles, num_shapes);

            vec3 base_colour = triangles[closest_intersection.index].material_.diffuse_light_component_;

            vec3 colour = monteCarlo(
                closest_intersection, 
                triangles,
                num_shapes, 
                seed,
                monte_carlo_samples
            );

            //colour = colour * vec3(base_colour.x / M_PI, base_colour.y / M_PI, base_colour.z / M_PI);

            //image[x * screen_width + y] = base_colour;
            //image[x * screen_width + y] = direct_light * base_colour + colour;
            image[x * screen_width + y] = colour;
        }
    }
}
//}

void draw(Camera & camera, Light & light, LightSphere & light_sphere, Triangle * triangles, int num_shapes, SdlWindowHelper sdl_window) {
    vec3 * image;
    cudaMallocManaged(&image, screen_height * screen_width * sizeof(vec3));

    int block_size = 256;
    int num_blocks = (screen_width + block_size - 1) / block_size;
    int seed = time(NULL);
    //draw_<<<num_blocks, block_size>>>(
    draw_<<<1,1>>>(
        camera, 
        light, 
        light_sphere, 
        triangles,
        num_shapes, 
        image, 
        screen_height, 
        screen_width, 
        focal_length, 
        seed,
        monte_carlo_samples
    );

    cudaDeviceSynchronize();

    renderImageBuffer(image, sdl_window);

    cudaFree(image);
}

void printVec3(vec3 v) {
    std::cout << v.x << "   " << v.y << "    " << v.z << std::endl;
}

__device__
vec3 monteCarlo(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris, 
    int seed,
    int monte_carlo_samples
) {
    vec3 indirect_light_approximation = vec3(0);
    int i = 0;
    int ctr = 0;
    while(i < monte_carlo_samples && ctr < monte_carlo_samples * 10) {
        Ray random_ray = Ray(
            closest_intersection.position, 
            closest_intersection.normal
        );

        vec3 random_point = randomPointInHemisphere(seed, closest_intersection.position, 1.0f);
        
        vec3 random_dir = random_point - vec3(random_ray.start_);
        random_dir = vec4(random_dir, 1);
 
        random_ray.direction_ = vec4(random_dir, 1);
        random_ray.start_ = (random_ray.start_ + 0.001f * random_ray.direction_);
        
        if (random_ray.closestIntersection(triangles, num_tris)) {
            Intersection indirect_light_intersection = random_ray.closest_intersection_;
            indirect_light_approximation = triangles[indirect_light_intersection.index].material_.diffuse_light_component_;

            ////////////////////
            float dist_point_to_light = glm::distance(
                closest_intersection.position, 
                indirect_light_intersection.position
            );
            
            vec3 surface_normal = vec3(closest_intersection.normal);
            
            vec3 surface_to_light_dir = vec3(
                indirect_light_intersection.position -
                closest_intersection.position
            );
            
            surface_to_light_dir = glm::normalize(surface_to_light_dir);
            Ray surface_to_light_ray(
                closest_intersection.position + 0.001f * vec4(surface_to_light_dir, 1),
                vec4(surface_to_light_dir, 1)
            );
            float scalar = (
                max(
                    dot(surface_to_light_dir, surface_normal), 
                    0.0f
                ) / (4.0f * M_PI * std::pow(dist_point_to_light, 2))
            );
            ///////////////////
            //printf("%f\n", scalar);
            indirect_light_approximation *= scalar;
            i++;
        }
        ctr += 1;
    }
    indirect_light_approximation /= monte_carlo_samples;


    return indirect_light_approximation;
}

__device__
vec3 randomPointInHemisphere(int seed, vec3 centre, float radius) {
    curandState_t state;

    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    int new_seed = id % 10 + seed;

    curand_init(new_seed, id, 0, &state);

    float max_val = radius;
    float min_val = -radius;
    //float min_val = 0;

    bool contained = false;
    while(!contained) {
        float rand_x = curand_uniform(&state);
        rand_x *= (max_val - min_val + 0.999999);
        rand_x += min_val;

        float rand_y = curand_uniform(&state);
        rand_y *= (max_val - min_val + 0.999999);
        rand_y += min_val;

        min_val = 0;
        float rand_z = curand_uniform(&state);
        rand_z *= (max_val - min_val + 0.999999);
        rand_z += min_val;
        
        vec3 random_point(
            centre.x + rand_x,
            centre.y + rand_y,
            centre.z + rand_z
        );
        if (glm::distance(random_point, centre) <= radius) {
            contained = true;
            return random_point;
        }
    }
    return vec3(0);
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
    Triangle floor_triangle_1 = Triangle(C, B, A, default_cyan);
    //triangles.push_back(floor_triangle_1);
    triangles[curr_tris] = floor_triangle_1;
    curr_tris++;

    Triangle floor_triangle_2 = Triangle(C, D, B, default_cyan);
    //triangles.push_back(floor_triangle_2);
    triangles[curr_tris] = floor_triangle_2;
    curr_tris++;

    // Left wall
    Triangle left_wall_1 = Triangle(A, E, C, default_yellow);
    //triangles.push_back(left_wall_1);
    triangles[curr_tris] = left_wall_1;
    curr_tris++;

    Triangle left_wall_2 = Triangle(C, E, G, default_yellow);
    //triangles.push_back(left_wall_2);
    triangles[curr_tris] = left_wall_2;
    curr_tris++;

    // Right wall
    Triangle right_wall_1 = Triangle(F, B, D, default_green);
    //triangles.push_back(right_wall_1);
    triangles[curr_tris] = right_wall_1;
    curr_tris++;

    Triangle right_wall_2 = Triangle(H, F, D, default_green);
    //triangles.push_back(right_wall_2);
    triangles[curr_tris] = right_wall_2;
    curr_tris++;

    // Ceiling
    Triangle ceiling_1 = Triangle(E, F, G, default_purple);
    //triangles.push_back(ceiling_1);
    triangles[curr_tris] = ceiling_1;
    curr_tris++;

    Triangle ceiling_2 = Triangle(F, H, G, default_purple);
    //triangles.push_back(ceiling_2);
    triangles[curr_tris] = ceiling_2;
    curr_tris++;

    // Back wall
    Triangle back_wall_1 = Triangle(G, D, C, default_white);
    //triangles.push_back(back_wall_1);
    triangles[curr_tris] = back_wall_1;
    curr_tris++;

    Triangle back_wall_2 = Triangle(G, H, D, default_white);
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
    //triangles.push_back(Triangle(E, B, A, default_red));
    triangles[curr_tris] = Triangle(E, B, A, default_red);
    curr_tris++;
    //triangles.push_back(Triangle(E, F, B, default_red));
    triangles[curr_tris] = Triangle(E, F, B, default_red);
    curr_tris++;

    // Front
    //triangles.push_back(Triangle(F, D, B, default_red));
    triangles[curr_tris] = Triangle(F, D, B, default_red);
    curr_tris++;
    //triangles.push_back(Triangle(F, H, D, default_red));
    triangles[curr_tris] = Triangle(F, H, D, default_red);
    curr_tris++;

    // BACK
    //triangles.push_back(Triangle(H, C, D, default_red));
    triangles[curr_tris] = Triangle(H, C, D, default_red);
    curr_tris++;
    //triangles.push_back(Triangle(H, G, C, default_red));
    triangles[curr_tris] = Triangle(H, G, C, default_red);
    curr_tris++;

    // LEFT
    //triangles.push_back(Triangle(G, E, C, default_red));
    triangles[curr_tris] = Triangle(G, E, C, default_red);
    curr_tris++;
    //triangles.push_back(Triangle(E, A, C, default_red));
    triangles[curr_tris] = Triangle(E, A, C, default_red);
    curr_tris++;

    // TOP
    //triangles.push_back(Triangle(G, F, E, default_red));
    triangles[curr_tris] = Triangle(G, F, E, default_red);
    curr_tris++;
    //triangles.push_back(Triangle(G, H, F, default_red));
    triangles[curr_tris] = Triangle(G, H, F, default_red);
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
   
    //triangles.push_back(Triangle(E, B, A, default_blue));
    triangles[curr_tris] = Triangle(E, B, A, default_blue);
    curr_tris++;
    //triangles.push_back(Triangle(E, F, B, default_blue));
    triangles[curr_tris] = Triangle(E, F, B, default_blue);
    curr_tris++;

    // Front
    //triangles.push_back(Triangle(F, D, B, default_blue));
    triangles[curr_tris] = Triangle(F, D, B, default_blue);
    curr_tris++;
    //triangles.push_back(Triangle(F, H, D, default_blue));
    triangles[curr_tris] = Triangle(F, H, D, default_blue);
    curr_tris++;

    // BACK
    //triangles.push_back(Triangle(H, C, D, default_blue));
    triangles[curr_tris] = Triangle(H, C, D, default_blue);
    curr_tris++;
    //triangles.push_back(Triangle(H, G, C, default_blue));
    triangles[curr_tris] = Triangle(H, G, C, default_blue);
    curr_tris++;

    // LEFT
    //triangles.push_back(Triangle(G, E, C, default_blue));
    triangles[curr_tris] = Triangle(G, E, C, default_blue);
    curr_tris++;
    //triangles.push_back(Triangle(E, A, C, default_blue));
    triangles[curr_tris] = Triangle(E, A, C, default_blue);
    curr_tris++;

    // TOP
    //triangles.push_back(Triangle(G, F, E, default_blue));
    triangles[curr_tris] = Triangle(G, F, E, default_blue);
    curr_tris++;
    //triangles.push_back(Triangle(G, H, F, default_blue));
    triangles[curr_tris] = Triangle(G, H, F, default_blue);
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
