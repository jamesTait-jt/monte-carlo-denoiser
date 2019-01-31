// This class is designed to ease the creation and rendering of windows in SDL2.
#ifndef SDL_WINDOW_HELPER
#define SDL_WINDOW_HELPER

#include <glm/glm.hpp>
#include <SDL2/SDL.h>

using glm::vec3;

// Creates a blank SDL_Window of specified width and height
// Example:
//     SdlWindowHelper sdl_window(640, 640);
//     sdl_window.putPixel(5, 5, colour)
//     sdl_window.render()
//     sdl_window.destroy()
class SdlWindowHelper {

    private:
        SDL_Window *window_;
        SDL_Renderer *renderer_;
        // Used to detect the closing of the sdl window
        SDL_Event event_;
        int window_width_;
        int window_height_;

    public:
        SdlWindowHelper(int width, int height);
        void putPixel(int x_pos, int y_pos, vec3 colour);
        void render();
        void destroy();
        bool noQuitMessage();
};

#endif
