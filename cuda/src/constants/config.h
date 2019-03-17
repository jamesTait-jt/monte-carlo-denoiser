#ifndef CONSTANTS_CONFIG_H
#define CONSTANTS_CONFIG_H

#include <vector>
#include <string>
#include <glm/glm.hpp>


using glm::vec3;
using glm::vec4;

// Screen configuration
const int SCREEN_WIDTH = 512; // The width of the final image
const int SCREEN_HEIGHT = 512; // The height of the final image

// Camera configurations
const vec4 camera_configurations[] = {
    vec4(0.0f, 0.0f, 1.0f, 1.0f), // Whole room open box
    //vec4(-0.99f, 0.1f, -0.99f, 1.0f), // Back left corner looking in (closed box)
    //vec4(0.99f, 0.1f, -0.99f, 1.0f),  // Back right corner looking in (closed box)
    vec4(-0.99f, 0.1f, 0.99f, 1.0f),  // Front left corner looking in (closed box)
    vec4(0.8f, 0.1f, 0.8f, 1.0f)    // Front right corner looking in (closed box)
};

const float camera_yaws[] = {
    0.0f,
    -(float) M_PI / 6.0f,
    (float) M_PI / 6.0f,
    //(float) 3 * M_PI / 4.0f,
    //-(float) 3 * M_PI / 4.0f
};

const std::string scenes[] = {
    "objects/james_room_closed_big_light.obj",
    "objects/james_room_closed_big_light.obj",
    "objects/james_room_closed_big_light.obj"
};

// Lights configuration
// -- lightSphere 1: 
const vec4 light_start_position(0.0f, -0.4f, -0.9f, 1.0f); // The start position of the centre of the area light
const float area_light_radius = 0.1f; // Distance away from the centre point lights can spawn
const int num_lights = 1; // The number of point lights in the area light
const float light_intensity = 0.5f; // The intensity of the area light as a whole
const vec3 light_colour(0.75f, 0.75f, 0.75f); // The colour of the area light

const int NUM_BOUNCES = 10; // Number of bounces in the path tracer before returning black

// Misc.
const float H_EPS = 0.0001f;

#endif
