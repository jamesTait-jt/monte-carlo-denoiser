#include <iostream>
#include <src/constants/config.h>
#include <vector>

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
    std::string extension = ".ppm";
    filename += (name + extension);
    FILE * file = fopen(filename.c_str(), "w");
    fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
    for(int i = 0 ; i < width * height; i++) {
        fprintf(file, "%d %d %d ", scaleTo255(image[i].x), 
                                   scaleTo255(image[i].y),
                                   scaleTo255(image[i].z)
        );
    }
    fclose(file);
    printf("Saved image to '%s'\n", filename.c_str());
}

void save_patches(vec3 * image, int size) {

    std::vector<std::vector<vec3>> image2d (
        screen_height,
        std::vector<vec3>(screen_width)
    );

    for(int i = 0 ; i < screen_width * screen_height ; i++) {
        int x = i / screen_width;
        int y = i % screen_width;
        image2d[x][y] = image[i];
    }

    printf("Saving...\n");

    int ctr = 0;
    for (int a = 0 ; a < screen_width - size ; a++) {
        for (int b = 0 ; b < screen_height - size ; b++) {
            ctr++;
            std::string filename = "out/patches/" + std::to_string(ctr) + ".ppm";
            FILE * patch = fopen(filename.c_str(), "wt");
            fprintf(patch, "P3\n%d %d\n%d\n", size, size, 255);
            for (int c = 0 ; c < size ; c++) {
                for (int d = 0 ; d < size ; d++) {
                    fprintf(patch, "%d %d %d ", scaleTo255(image2d[a + c][b + d].x),
                                                scaleTo255(image2d[a + c][b + d].y),
                                                scaleTo255(image2d[a + c][b + d].z)
                    );
                }
            }
            fclose(patch);
        }
    }
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

// This function creates a new coordinate system in which the up vector is
// oriented along the shaded point normal
__device__
void createCoordinateSystem(const vec3 & N, vec3 & N_t, vec3 & N_b) {
    if (std::fabs(N.x) > std::fabs(N.y)) {
        N_t = vec3(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
    } else {
        N_t = vec3(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
    }
    N_b = glm::cross(N, N_t);
}

// Given two random numbers between 0 and 1, return a direction to a point on a
// hemisphere
__device__
vec3 uniformSampleHemisphere(const float & r1, const float & r2) {
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sin_theta = sqrtf(1 - r1 * r1);
    float phi = 2 * (float)M_PI * r2;
    float x = sin_theta * cosf(phi);
    float z = sin_theta * sinf(phi);
    return vec3(x, r1, z);
}

// Calculates the mean of a data set of vec3
__device__
vec3 mean(vec3 * data, int data_size) {
    vec3 accum(0.0f);
    for (int i = 0 ; i < data_size ; i++) {
       accum += data[i];
    }
    return accum / (float) data_size;
}

__device__
float mySum(vec3 v) {
    return v.x + v.y + v.z;
}
