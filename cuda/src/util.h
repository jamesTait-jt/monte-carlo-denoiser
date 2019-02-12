#ifndef UTIL_H
#define UTIL_H

#include <glm/glm.hpp>

using glm::vec3;

float clamp(
    float x
);

int scaleTo255(
    float x
);

void save_image(
    vec3 * image,
    int height,
    int width,
    const char * name
);

#endif
