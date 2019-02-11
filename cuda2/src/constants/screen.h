#ifndef CONSTANTS_SCREEN_H
#define CONSTANTS_SCREEN_H

const int screen_width = 256;
const int screen_height = 256;
const int supersample_width = screen_width * 4;
const int supersample_height = screen_height * 4;
const int focal_length = supersample_height;

const int monte_carlo_samples = 128;
const int monte_carlo_depth = 2;

#endif
