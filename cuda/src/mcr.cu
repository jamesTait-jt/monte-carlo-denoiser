#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <omp.h>

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

    Camera camera(vec4(0, 0, -3, 1));
    Light light(10.0f, vec3(1), vec4(0, -0.4, -0.9, 1.0));
    LightSphere light_sphere(vec4(0, -0.4, -0.9, 1.0), 0.1f, 5, 10.0f, vec3(1));
    
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
void draw_(Camera camera, Light & light, LightSphere light_sphere, Triangle * triangles, int num_shapes, vec3 * image, int screen_height, int screen_width, int focal_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index ; i < screen_height * screen_width ; i += stride) {
        int x = i / screen_height;
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
            vec3 colour = monteCarlo(closest_intersection, triangles, num_shapes);

            //image[x * screen_width + y] = base_colour;
            image[x * screen_width + y] = direct_light * base_colour;
        }
    }
}
//}

void draw(Camera & camera, Light & light, LightSphere & light_sphere, Triangle * triangles, int num_shapes, SdlWindowHelper sdl_window) {
    vec3 * image;
    cudaMallocManaged(&image, screen_height * screen_width * sizeof(vec3));

    int block_size = 256;
    int num_blocks = (screen_width + block_size - 1) / block_size;
    draw_<<<num_blocks, block_size>>>(camera, light, light_sphere, triangles, num_shapes, image, screen_height, screen_width, focal_length);

    cudaDeviceSynchronize();

    renderImageBuffer(image, sdl_window);

    cudaFree(image);
}

void printVec3(vec3 v) {
    std::cout << v.x << "   " << v.y << "    " << v.z << std::endl;
}

__device__
vec3 monteCarlo(Intersection closest_intersection, Triangle * triangles, int num_tris) {
    vec3 indirect_light_approximation = vec3(0);
    int i = 0;
    while(i < monte_carlo_samples) {
        Ray random_ray = Ray(
            closest_intersection.position, 
            closest_intersection.normal
        );
        int idx = threadIdx.x+blockDim.x*blockIdx.x;
        float rand_theta = curand_uniform( * M_PI;
        random_ray.rotateRay(rand_theta);
        random_ray.start_ = (random_ray.start_ + 0.001f * random_ray.direction_);
        
        if (random_ray.closestIntersection(triangles, num_tris)) {
           Intersection indirect_light_intersection = random_ray.closest_intersection_;
           indirect_light_approximation =
               triangles[indirect_light_intersection.index].material_.diffuse_light_component_;
           i++;
        }
        
    }
    indirect_light_approximation /= monte_carlo_samples;
    return indirect_light_approximation;
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
