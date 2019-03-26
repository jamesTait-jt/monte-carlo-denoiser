#include <vector>
#include <string>

#include "triangle.h"
#include "sphere.h"
#include "camera.h"
#include "light.h"
#include "lightSphere.h"
#include "sdl_window_helper.h"

using glm::vec3;

// File names
std::string pre_alias_title = "pre_alias"; // Name of saved image before aliasing
std::string aliased_title = "aliased"; // Name of the saved image after aliasing

void loadShapes(
    Triangle * triangles,
    Sphere * spheres
);

void update(
    Camera & camera, 
    Light & light
);

void draw(
    Camera & camera, 
    Light & light, 
    LightSphere & light_sphere, 
    Triangle * triangles, 
    int num_shapes, 
    SdlWindowHelper sdl_helper
);

__global__
void render_init(
    curandState * rand_state,
    int seed
);

__global__
void render_kernel(
    vec3 * colours,
    vec3 * surface_normals,
    vec3 * albedos,
    float * depths,
    vec3 * colour_variances,
    vec3 * surface_normal_variances,
    vec3 * albedo_variances,
    float * depth_variances,
    Camera camera,
    Triangle * triangles,
    int num_tris,
    Sphere * spheres,
    int num_spheres,
    curandState * rand_state,
    bool is_reference_image
);

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
);

void generateCameraStartPositions(
    vec4 * camera_start_posiions,
    float * camera_start_yaws
);

void view_live(
    vec3 * image,
    SdlWindowHelper sdl_helper
);
