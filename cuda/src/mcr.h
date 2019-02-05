#include <vector>

#include "triangle.h"
#include "camera.h"
#include "light.h"
#include "lightSphere.h"
#include "sdl_window_helper.h"

using glm::vec3;

void loadShapes(Triangle * triangles);
void update(Camera & camera, Light & light);
void draw(Camera & camera, Light & light, LightSphere & light_sphere, Triangle * triangles, int num_shapes, SdlWindowHelper sdl_helper);
__device__ vec3 monteCarlo(Intersection closest_intersection, Triangle * triangles, int num_tris);
void renderImageBuffer(vec3 * image, SdlWindowHelper sdl_window);

void printVec3(vec3 v);
