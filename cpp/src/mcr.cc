#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include "glm/gtx/string_cast.hpp"
#include <SDL2/SDL.h>
#include <omp.h>

#include <iostream>

#include "constants/screen.h"
#include "constants/materials.h"
#include "mcr.h"
#include "shape.h"
#include "triangle.h"

int main (int argc, char* argv[]) {

    omp_set_num_threads(4);
    
    // Seed for random numbers
    srand (time(NULL));

    std::vector<Triangle> triangles;

    loadShapes(triangles);

    std::vector<Shape *> shapes;
    int num_tris = triangles.size();
    for (int i = 0 ; i < num_tris ; i++) {
        Shape * shape_pointer (&triangles[i]);
        shapes.push_back(shape_pointer);
    }

    Camera camera(vec4(0, 0, -3, 1));
    Light light(10.0f, vec3(1), vec4(0, -0.4, -0.9, 1.0));
    LightSphere light_sphere(vec4(0, -0.4, -0.9, 1.0), 0.1f, 1, 10.0f, vec3(1));
    
    SdlWindowHelper sdl_window(SCREEN_WIDTH, SCREEN_HEIGHT);
    
    int i = 0;
    while(sdl_window.noQuitMessage() && i < 2) {
        i++;
        update(camera, light);
        draw(camera, light, light_sphere, shapes, sdl_window);
    }

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

void draw(Camera & camera, Light & light, LightSphere & light_sphere, std::vector<Shape *> shapes, SdlWindowHelper sdl_window) {
    std::vector<std::vector<vec3>> image(
        SCREEN_HEIGHT,
        std::vector<vec3>(SCREEN_HEIGHT)
    );
    #pragma omp parallel for
    for (int x = 0 ; x < SCREEN_HEIGHT ; x++) {
        for (int y = 0 ; y < SCREEN_WIDTH ; y++) {
            // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
            vec4 dir((x - SCREEN_WIDTH / 2) , (y - SCREEN_HEIGHT / 2) , focal_length , 1);
            
            // Create a ray that we will change the direction for below
            Ray ray(camera.position_, dir);
            ray.rotateRay(camera.yaw_);

            if (ray.closestIntersection(shapes)) {
                Intersection closest_intersection = ray.closest_intersection_;
                vec3 colour = monteCarlo(
                    closest_intersection, 
                    shapes,
                    light_sphere,
                    monte_carlo_depth,
                    0
                );
                image[x][y] =  colour;
            }
        }
    }
    renderImageBuffer(image, sdl_window);
}

// Calculates the indirect and direct light estimation for diffuse objects
vec3 monteCarlo(
    Intersection closest_intersection, 
    std::vector<Shape *> shapes, 
    LightSphere light_sphere,
    int max_depth,
    int depth
) {
    if (depth >= max_depth) {
        vec3 direct_light = light_sphere.directLight(
            closest_intersection,
            shapes
        );
        vec3 base_colour = shapes[closest_intersection.index]->material_.diffuse_light_component_;
        return direct_light * base_colour;
    }

    vec3 intersection_normal_3 = vec3(closest_intersection.normal);
    vec3 base_colour = shapes[closest_intersection.index]->material_.diffuse_light_component_;
    vec3 direct_light = light_sphere.directLight(
        closest_intersection,
        shapes 
    );

    vec3 N_t, N_b;
    createCoordinateSystem(intersection_normal_3, N_t, N_b);

    vec3 indirect_estimate = vec3(0);
    float pdf = 1 / (2 * M_PI);
    for (int i = 0 ; i < monte_carlo_samples ; i++) {
        float r1 = rand() / (float) RAND_MAX; // cos(theta) = N.Light Direction
        float r2 = rand() / (float) RAND_MAX;
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

        if (random_ray.closestIntersection(shapes)) {
               indirect_estimate += r1 * monteCarlo(
               random_ray.closest_intersection_,
               shapes,
               light_sphere,
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
void createCoordinateSystem(const vec3 & N, vec3 & N_t, vec3 & N_b) {
    if (std::fabs(N.x) > std::fabs(N.y)) {
        N_t = vec3(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
    } else {
        N_t = vec3(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
    }
    N_b = glm::cross(N, N_t);
} 

void renderImageBuffer(std::vector<std::vector<vec3>> image, SdlWindowHelper sdl_window) {
    for (int x = 0 ; x < SCREEN_HEIGHT ; x++) {
        for (int y = 0 ; y < SCREEN_WIDTH ; y++) {
           sdl_window.putPixel(x, y, image[x][y]); 
        }
    }
    sdl_window.render();
}

void loadShapes(std::vector<Triangle> & triangles) {
    float cornell_length = 555;			// Length of Cornell Box side.

    vec4 A(cornell_length, 0, 0             , 1);
    vec4 B(0             , 0, 0             , 1);
    vec4 C(cornell_length, 0, cornell_length, 1);
    vec4 D(0             , 0, cornell_length, 1);

    vec4 E(cornell_length, cornell_length, 0             , 1);
    vec4 F(0             , cornell_length, 0             , 1);
    vec4 G(cornell_length, cornell_length, cornell_length, 1);
    vec4 H(0             , cornell_length, cornell_length, 1);

    // Triangles now take a material as an argument rather than a colour
    // Floor:
    Triangle floor_triangle_1 = Triangle(C, B, A, default_cyan);
    triangles.push_back(floor_triangle_1);

    Triangle floor_triangle_2 = Triangle(C, D, B, default_cyan);
    triangles.push_back(floor_triangle_2);

    // Left wall
    Triangle left_wall_1 = Triangle(A, E, C, default_yellow);
    triangles.push_back(left_wall_1);

    Triangle left_wall_2 = Triangle(C, E, G, default_yellow);
    triangles.push_back(left_wall_2);

    // Right wall
    Triangle right_wall_1 = Triangle(F, B, D, default_green);
    triangles.push_back(right_wall_1);

    Triangle right_wall_2 = Triangle(H, F, D, default_green);
    triangles.push_back(right_wall_2);

    // Ceiling
    Triangle ceiling_1 = Triangle(E, F, G, default_purple);
    triangles.push_back(ceiling_1);

    Triangle ceiling_2 = Triangle(F, H, G, default_purple);
    triangles.push_back(ceiling_2);

    // Back wall
    Triangle back_wall_1 = Triangle(G, D, C, default_white);
    triangles.push_back(back_wall_1);

    Triangle back_wall_2 = Triangle(G, H, D, default_white);
    triangles.push_back(back_wall_2);

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
    triangles.push_back(Triangle(E, B, A, default_red));
    triangles.push_back(Triangle(E, F, B, default_red));

    // Front
    triangles.push_back(Triangle(F, D, B, default_red));
    triangles.push_back(Triangle(F, H, D, default_red));

    // BACK
    triangles.push_back(Triangle(H, C, D, default_red));
    triangles.push_back(Triangle(H, G, C, default_red));

    // LEFT
    triangles.push_back(Triangle(G, E, C, default_red));
    triangles.push_back(Triangle(E, A, C, default_red));

    // TOP
    triangles.push_back(Triangle(G, F, E, default_red));
    triangles.push_back(Triangle(G, H, F, default_red));

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
   
    triangles.push_back(Triangle(E, B, A, default_blue));
    triangles.push_back(Triangle(E, F, B, default_blue));

    // Front
    triangles.push_back(Triangle(F, D, B, default_blue));
    triangles.push_back(Triangle(F, H, D, default_blue));

    // BACK
    triangles.push_back(Triangle(H, C, D, default_blue));
    triangles.push_back(Triangle(H, G, C, default_blue));

    // LEFT
    triangles.push_back(Triangle(G, E, C, default_blue));
    triangles.push_back(Triangle(E, A, C, default_blue));

    // TOP
    triangles.push_back(Triangle(G, F, E, default_blue));
    triangles.push_back(Triangle(G, H, F, default_blue));

    // ----------------------------------------------
    // Scale to the volume [-1,1]^3

    for (size_t i = 0 ; i < triangles.size() ; ++i) {
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
        triangles[i].v0_ = new_v0;

        vec4 new_v1 = triangles[i].v1_;
        new_v1.x *= -1;
        new_v1.y *= -1;
        new_v1.w = 1.0;
        triangles[i].v1_ = new_v1;

        vec4 new_v2 = triangles[i].v2_;
        new_v2.x *= -1;
        new_v2.y *= -1;
        new_v2.w = 1.0;
        triangles[i].v2_ = new_v2;

        triangles[i].computeAndSetNormal();
    }
}
