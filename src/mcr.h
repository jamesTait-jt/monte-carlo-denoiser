#include <vector>

#include "triangle.h"
#include "camera.h"
#include "sdl_window_helper.h"

using glm::vec3;

void loadShapes(std::vector<Triangle> & triangles);
void update(Camera & camera);
void draw(Camera & camera, std::vector<Shape *> shapes, SdlWindowHelper sdl_helper);
void renderImageBuffer(std::vector<std::vector<vec3>> image, SdlWindowHelper sdl_window);
