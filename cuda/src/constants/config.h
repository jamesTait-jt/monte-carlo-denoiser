#ifndef CONSTANTS_CONFIG_H
#define CONSTANTS_CONFIG_H

#include <glm/glm.hpp>

using glm::vec3;
using glm::vec4;

// Screen configuration
const int screen_width = 512; // The width of the final image
const int screen_height = 512; // The height of the final image
const int anti_aliasing_factor = 1; // Factor by which we multiply the image dimensions to get the size of the supersampled image

// Camera configuration
const vec4 cam_start_position(0.0f, 0.0f, -3.0f, 1.0f); // The starting position of the camera
const float cam_start_yaw = 0; // Starting direction of the camera
const int cam_focal_length = screen_height * anti_aliasing_factor; // Focal length of the camera must match the supersampled dimensions

// Lights configuration
// -- lightSphere 1: 
const vec4 light_start_position(0.0f, -0.4f, -0.9f, 1.0f); // The start position of the centre of the area light
const float area_light_radius = 0.1f; // Distance away from the centre point lights can spawn
const int num_lights = 1; // The number of point lights in the area light
const float light_intensity = 12.0f; // The intensity of the area light as a whole
const vec3 light_colour(0.75f, 0.75f, 0.75f); // The colour of the area light

// Monte carlo configuration
const int monte_carlo_max_depth = 1; // Number of bounces in the monte carlo estimation
const int monte_carlo_num_samples = 128; // Number of samples per pixel in monte carlo estimation

// Misc.
const float float_precision_error = 0.0001f;

#endif
