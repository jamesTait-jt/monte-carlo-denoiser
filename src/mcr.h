#include <vector>

#include "triangle.h"
#include "camera.h"
#include "light.h"
#include "sdl_window_helper.h"

using glm::vec3;

void loadShapes(std::vector<Triangle> & triangles);
void update(Camera & camera, Light & light);
void draw(Camera & camera, Light & light, std::vector<Shape *> shapes, SdlWindowHelper sdl_helper);
vec3 monteCarlo(Intersection closest_intersection, std::vector<Shape *> shapes);
void renderImageBuffer(std::vector<std::vector<vec3>> image, SdlWindowHelper sdl_window);
