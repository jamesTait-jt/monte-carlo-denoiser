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

__device__
void createCoordinateSystem(
    const vec3 & N, 
    vec3 & N_t, 
    vec3 & N_b
);

__device__ 
vec3 monteCarlo2(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris, 
    LightSphere light_sphere,
    int seed,
    int monte_carlo_samples,
    int max_depth,
    int depth
);

__device__ 
vec3 monteCarlo(
    Intersection closest_intersection, 
    Triangle * triangles, 
    int num_tris,
    int seed,
    int monte_carlo_samples
);

__device__
float uniform_rand(
    int seed
);

__device__
vec3 randomPointInHemisphere(
    int seed,
    vec3 centre,
    float radius
);

void renderImageBuffer(
    vec3 * image,
    SdlWindowHelper sdl_window
);

void printVec3(
    vec3 v
);
