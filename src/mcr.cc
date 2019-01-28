#include <glm/glm.hpp>
#include <SDL2/SDL.h>

#include <iostream>

#include "sdl_window_helper.h"

int main (int argc, char* argv[]) {

    // Define the height and width of the image 
    int window_height = 1000;
    int window_width = 1000;

    SdlWindowHelper sdl_window(window_width, window_height);
    for (int i = 0 ; i < window_height ; ++i)
        if (i % 2 == 0)
            sdl_window.putPixel(i, i, vec3(100, 100, 100));

    sdl_window.renderUntilExit();

    return 0;
}
