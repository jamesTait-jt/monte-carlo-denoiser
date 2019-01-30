#include "sdl_window_helper.h"
#include <algorithm>

// Constructs the sdl window and renderer, then renders a blank image
SdlWindowHelper::SdlWindowHelper(int width, int height) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(width, height, 0, &this->window_, &this->renderer_);
    SDL_SetRenderDrawColor(this->renderer_, 0, 0, 0, 0);
    SDL_RenderClear(this->renderer_);
}

// Draws a pixel at a given position with a given colour 
//
// Params:
//     x_pos: The x position of the pixel
//     y_pos: The y position of the pixel
//     colour: The colour that the pixel should be drawn
void SdlWindowHelper::putPixel(int x_pos, int y_pos, vec3 colour) {
    colour.x = std::max(0.0, std::min(1.0, (double)colour.x));
    colour.x = floor(colour.x == 1.0 ? 255 : colour.x * 256.0);
    
    colour.y = std::max(0.0, std::min(1.0, (double)colour.y));
    colour.y = floor(colour.y == 1.0 ? 255 : colour.y * 256.0);
    
    colour.z = std::max(0.0, std::min(1.0, (double)colour.z));
    colour.z = floor(colour.z == 1.0 ? 255 : colour.z * 256.0);

    SDL_SetRenderDrawColor(this->renderer_, colour.x, colour.y, colour.z, 255);
    SDL_RenderDrawPoint(this->renderer_, x_pos, y_pos);
}

// Renders the image, waits until the user closes the window and then destroys
// associated sdl objects.
void SdlWindowHelper::render() {
    SDL_RenderPresent(this->renderer_);
    while (1) {
        if (SDL_PollEvent(&this->event_) && this->event_.type == SDL_QUIT)
            break;
    }
    this->destroy();
}

// Destroys SDL objects and cleans up with Quit()
void SdlWindowHelper::destroy() {
    SDL_DestroyRenderer(this->renderer_);
    SDL_DestroyWindow(this->window_);
    SDL_Quit();
}
