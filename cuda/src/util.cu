#include <iostream>

#include "util.h"

// Clamps a float x between 0 and 1
float clamp(float x) { 
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; 
} 

// convert RGB float in range [0,1] to int in range [0, 255]
int scaleTo255(float x) {
    return int(clamp(x) * 255); 
}

// Saves the image as a simple .ppm file (open with program such as 'feh'
void save_image(vec3 * image, int height, int width, std::string name) {
    std::string filename = "out/";
    filename += name;
    FILE * file = fopen(filename.c_str(), "w");
    fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
    for(int i = 0 ; i < width * height; i++) {
        fprintf(file, "%d %d %d ", scaleTo255(image[i].x), 
                                   scaleTo255(image[i].y),
                                   scaleTo255(image[i].z)
        );
    }
    printf("Saved image to '%s'\n", filename.c_str());
}

float maxf(float a, float b) {
    return a > b ? a : b;
}

__host__ __device__
void swap(float & a, float & b) {
    float temp = a;
    a = b;
    b = temp;
}
