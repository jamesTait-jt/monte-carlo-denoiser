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
    
    SdlWindowHelper sdl_window(screen_width, screen_height);
    
    int i = 0;
    while(sdl_window.noQuitMessage() && i < 1000) {
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
        screen_height,
        std::vector<vec3>(screen_height)
    );
    //#pragma omp parallel for
    for (int x = 0 ; x < screen_height ; x++) {
        for (int y = 0 ; y < screen_width ; y++) {
            // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
            vec4 dir((x - screen_width / 2) , (y - screen_height / 2) , focal_length , 1);
            
            // Create a ray that we will change the direction for below
            Ray ray(camera.get_position(), dir);
            ray.rotateRay(camera.get_yaw());

            if (ray.closestIntersection(shapes)) {
                Intersection closest_intersection = ray.get_closest_intersection();
                //vec3 direct_light = light.directLight(closest_intersection, shapes);
                vec3 direct_light = light_sphere.directLight(closest_intersection, shapes);
                vec3 base_colour = shapes[closest_intersection.index]->get_material().get_diffuse_light_component();
                direct_light *= base_colour;
                //vec3 colour = monteCarlo(closest_intersection, shapes);
                //image[x][y] = direct_light * colour;
                image[x][y] = direct_light;
            }
        }
    }
    renderImageBuffer(image, sdl_window);
}

vec3 monteCarlo(Intersection closest_intersection, std::vector<Shape *> shapes) {
    vec3 indirect_light_approximation = vec3(0);
    int i = 0;
    while(i < monte_carlo_samples) {
        Ray random_ray = Ray(
            closest_intersection.position, 
            closest_intersection.normal
        );
        float rand_theta = drand48() * M_PI;
        random_ray.rotateRay(rand_theta);
        random_ray.set_start(random_ray.get_start() + 0.001f * random_ray.get_direction());

        if (random_ray.closestIntersection(shapes)) {
           Intersection indirect_light_intersection = random_ray.get_closest_intersection();
           indirect_light_approximation = shapes[indirect_light_intersection.index]->get_material().get_diffuse_light_component();
           i++;
        }
    }
    indirect_light_approximation /= monte_carlo_samples;
    return indirect_light_approximation;
}

void renderImageBuffer(std::vector<std::vector<vec3>> image, SdlWindowHelper sdl_window) {
    for (int x = 0 ; x < screen_height ; x++) {
        for (int y = 0 ; y < screen_width ; y++) {
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
