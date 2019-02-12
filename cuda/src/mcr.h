#include <vector>

#include "triangle.h"
#include "camera.h"
#include "light.h"
#include "lightSphere.h"
#include "sdl_window_helper.h"

using glm::vec3;

void loadShapes(
    Triangle * triangles
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
    curandState * rand_state
);

__global__
void render_kernel(
    vec3 * output,
    Camera camera,
    LightSphere light_sphere,
    Triangle * triangles,
    int num_tris,
    curandState * rand_state
);

__device__ 
vec3 monteCarlo(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris,
    LightSphere light_sphere,
    curandState rand_state,
    int max_depth,
    int depth
);

__device__
vec3 indirectLight(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris,
    LightSphere light_sphere,
    curandState rand_state,
    int max_depth,
    int depth
);

__device__
vec3 uniformSampleHemisphere(
    const float & r1, 
    const float & r2
);

__device__
void createCoordinateSystem(
    const vec3 & N, 
    vec3 & N_t, 
    vec3 & N_b
);

__global__
void MSAA(
    vec3 * supersampled_image, 
    vec3 * aliased_output
);

