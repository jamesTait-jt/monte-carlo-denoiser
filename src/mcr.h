#include <vector>

#include "triangle.h"
#include "camera.h"
#include "sdl_window_helper.h"

void loadShapes(std::vector<Triangle> & triangles);
void update(Camera & camera);
void draw(Camera & camera, std::vector<Shape *> shapes, SdlWindowHelper sdl_helper);
