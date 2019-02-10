#include <vector>

#include "triangle.h"
#include "camera.h"
#include "light.h"
#include "lightSphere.h"
#include "sdl_window_helper.h"

using glm::vec3;

void loadShapes(
    std::vector<Triangle> & triangles
);

void update(
    Camera & camera, 
    Light & light
);

void draw(
    Camera & camera, 
    Light & light, 
    LightSphere & light_sphere, 
    std::vector<Shape *> shapes, 
    SdlWindowHelper sdl_helper
);

vec3 monteCarlo(
    Intersection closest_intersection, 
    std::vector<Shape *> shapes, 
    LightSphere light_sphere,
    int max_depth,
    int depth
);

vec3 uniformSampleHemisphere(
    const float & r1, 
    const float & r2
);

void createCoordinateSystem(
    const vec3 & N, 
    vec3 & N_t, 
    vec3 & N_b
);

void renderImageBuffer(
    std::vector<std::vector<vec3>> image, 
    SdlWindowHelper sdl_window
);
