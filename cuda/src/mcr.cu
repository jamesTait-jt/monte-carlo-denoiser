#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <omp.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "constants/config.h"
#include "constants/materials.h"
#include "mcr.h"
#include "triangle.h"
#include "sphere.h"
#include "util.h"
#include "objectImporter.h"
#include "ray.cuh"


// ----- DEVICE CONSTANTS ----- //
//__constant__ int D_REF_SAMPLES_PER_PIXEL = 16 * 1024;
//__constant__ int D_NOISY_SAMPLES_PER_PIXEL = 3 * 256;
__constant__ int D_REF_SAMPLES_PER_PIXEL = 65536;
__constant__ int D_NOISY_SAMPLES_PER_PIXEL = 128;

// The error for floating point arithmetic issues
__constant__ float D_EPS = 0.0001f;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

int main(int argc, char *argv[]) {

    // Randomly generate the numbers of each file so that the data is shuffled
    std::vector<int> seed;
    for (int i = 1; i <= 226; i++) {
        seed.push_back(i);
    }

    std::random_shuffle(seed.begin(), seed.end());

    // Generate the material map
    std::unordered_map<std::string, Material> material_map = {
            {"base03",    m_sol_base03},
            {"base02",    m_sol_base02},
            {"base01",    m_sol_base01},
            {"base00",    m_sol_base00},
            {"base0",     m_sol_base0},
            {"base1",     m_sol_base1},
            {"base2",     m_sol_base2},
            {"base3",     m_sol_base3},
            {"yellow",    m_sol_yellow},
            {"orange",    m_sol_orange},
            {"red",       m_sol_red},
            {"magenta",   m_sol_magenta},
            {"violet",    m_sol_violet},
            {"blue",      m_sol_blue},
            {"cyan",      m_sol_cyan},
            {"green",     m_sol_green},
            {"light",     m_light},
            {"def_red"  , m_red},
            {"def_green", m_green},
    };

    int total_scenes = 1; //scenes.size();
    for (int scene_index = 0; scene_index < total_scenes; scene_index++) {

        Light light(light_intensity, halogen_light_colour, light_start_position);
        LightSphere ls(light_start_position, 0.5, 100, light_intensity, halogen_light_colour);

        for (int i = 0; i < 1; i++) {

            bool is_reference_image = i == 0 ? true : false;

            // ----- IMAGE ----- //

            // Pointer to the image on the host (CPU)
            vec3 *h_colours = new vec3[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the image on the device (GPU)
            vec3 *d_colours;

            // ----- FEATURE BUFFERS ----- //

            // Pointer to the surface normals on host
            vec3 *h_surface_normals = new vec3[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the surface normals on device
            vec3 *d_surface_normals;

            // Pointer to the albedo buffer on host
            vec3 *h_albedos = new vec3[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the albedo buffer on the device
            vec3 *d_albedos;

            // Pointer to the depth buffer on host
            float *h_depths = new float[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the depth buffer on device
            float *d_depths;

            // ----- VARIANCES ----- //

            // Pointer to the colour variances on host
            vec3 *h_colour_variances = new vec3[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the colour variances on device
            vec3 *d_colour_variances;

            // Pointer to the surface normals on host
            vec3 *h_surface_normal_variances = new vec3[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the surface normals on device
            vec3 *d_surface_normal_variances;

            // Pointer to the albedo buffer on host
            vec3 *h_albedo_variances = new vec3[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the albedo buffer on the device
            vec3 *d_albedo_variances;

            // Pointer to the depth buffer on host
            float *h_depth_variances = new float[SCREEN_WIDTH * SCREEN_HEIGHT];
            // Pointer to the depth buffer on device
            float *d_depth_variances;

            // Allocate memory on CUDA device
            checkCudaErrors(cudaMalloc(&d_colours, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3)));
            checkCudaErrors(cudaMalloc(&d_surface_normals, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3)));
            checkCudaErrors(cudaMalloc(&d_albedos, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3)));
            checkCudaErrors(cudaMalloc(&d_depths, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float)));

            checkCudaErrors(cudaMalloc(&d_colour_variances, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3)));
            checkCudaErrors(cudaMalloc(&d_surface_normal_variances, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3)));
            checkCudaErrors(cudaMalloc(&d_albedo_variances, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3)));
            checkCudaErrors(cudaMalloc(&d_depth_variances, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float)));

            // Specify the block and grid dimensions to schedule CUDA threads
            dim3 threads_per_block(8, 8);
            dim3 num_blocks(
                    SCREEN_WIDTH / threads_per_block.x,
                    SCREEN_HEIGHT / threads_per_block.y
            );

            // Create a vector of random states for use on the device
            curandState * d_rand_states;
            checkCudaErrors(cudaMalloc((void **) &d_rand_states, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(curandState)));

            // Load in the shapes
            std::vector<Triangle> triangles_vec;
            //printf(scenes[scene_index].c_str());
            load_scene(scenes[scene_index].c_str(), triangles_vec, material_map);

            int num_tris = triangles_vec.size();
            //int num_tris = 30;
            Triangle *triangles = new Triangle[num_tris];

            for (int j = 0; j < num_tris; j++) {
                triangles[j] = triangles_vec[j];
            }

            int num_spheres = 0;
            Sphere * spheres = new Sphere[num_spheres];

            // Load the polygons into the triangles array
            //loadShapes(triangles, spheres);

            Triangle *d_triangles;
            Sphere * d_spheres;

            checkCudaErrors(cudaMalloc(&d_triangles, num_tris * sizeof(Triangle)));
            checkCudaErrors(cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)));


            checkCudaErrors(cudaMemcpy(
                    d_triangles,
                    triangles,
                    num_tris * sizeof(Triangle),
                    cudaMemcpyHostToDevice
            ));

            checkCudaErrors(cudaMemcpy(
                    d_spheres,
                    spheres,
                    num_spheres * sizeof(Sphere),
                    cudaMemcpyHostToDevice
            ));

            printf("CUDA has been initialised. Begin rendering...\n");
            printf("=============================================\n\n");

            // Initialise the camera object
            Camera camera(
                camera_configurations[scene_index],
                camera_yaws[scene_index],
                SCREEN_HEIGHT
            );

            auto start = std::chrono::high_resolution_clock::now();

            int seed = time(NULL);
            // Launch the CUDA kernel from the host and begin rendering
            render_init << < num_blocks, threads_per_block >> > (
                d_rand_states,
                seed
            );

            // Render the reference image
            render_kernel<<<num_blocks, threads_per_block>>> (
                d_colours,
                d_surface_normals,
                d_albedos,
                d_depths,
                d_colour_variances,
                d_surface_normal_variances,
                d_albedo_variances,
                d_depth_variances,
                camera,
                d_triangles,
                num_tris,
                d_spheres,
                num_spheres,
                d_rand_states,
                is_reference_image,
                light,
                ls
            );

            // Copy results of rendering back to the host
            checkCudaErrors(cudaMemcpy(
                    h_colours,
                    d_colours,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_colour_variances,
                    d_colour_variances,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_surface_normals,
                    d_surface_normals,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_surface_normal_variances,
                    d_surface_normal_variances,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_albedos,
                    d_albedos,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_albedo_variances,
                    d_albedo_variances,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_depths,
                    d_depths,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float),
                    cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                    h_depth_variances,
                    d_depth_variances,
                    SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float),
                    cudaMemcpyDeviceToHost
            ));

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = end - start;
            int duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

            printf("Finished rendering in %dms.\n", duration_in_ms);

            std::string title_prefix = is_reference_image ? "reference_" : "noisy_";
            title_prefix = directory_names[scene_index] + "/" + title_prefix;
            save_image(
                    h_colours,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "colour_" + std::to_string(scene_index)
            );

            save_image(
                    h_colour_variances,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "colour_vars_" + std::to_string(scene_index)
            );

            save_image(
                    h_surface_normals,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "sn_" + std::to_string(scene_index)
            );

            save_image(
                    h_surface_normal_variances,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "sn_vars_" + std::to_string(scene_index)
            );

            save_image(
                    h_albedos,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "albedo_" + std::to_string(scene_index)
            );

            save_image(
                    h_albedo_variances,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "albedo_vars_" + std::to_string(scene_index)
            );

            save_image(
                    h_depths,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "depth_" + std::to_string(scene_index)
            );

            save_image(
                    h_depth_variances,
                    SCREEN_HEIGHT,
                    SCREEN_WIDTH,
                    title_prefix + "depth_vars_" + std::to_string(scene_index)
            );

            // Free CUDA memory
            checkCudaErrors(cudaFree(d_rand_states));

            checkCudaErrors(cudaFree(d_colours));
            checkCudaErrors(cudaFree(d_surface_normals));
            checkCudaErrors(cudaFree(d_albedos));
            checkCudaErrors(cudaFree(d_depths));

            checkCudaErrors(cudaFree(d_colour_variances));
            checkCudaErrors(cudaFree(d_surface_normal_variances));
            checkCudaErrors(cudaFree(d_albedo_variances));
            checkCudaErrors(cudaFree(d_depth_variances));

            checkCudaErrors(cudaFree(d_triangles));
            checkCudaErrors(cudaFree(d_spheres));

            // Clear memory for host
            delete[] h_colours;
            delete[] h_surface_normals;
            delete[] h_albedos;
            delete[] h_depths;

            delete[] h_colour_variances;
            delete[] h_surface_normal_variances;
            delete[] h_albedo_variances;
            delete[] h_depth_variances;

            delete[] triangles;
            delete[] spheres;
        }
    }
    return 0;
}

void view_live(
    vec3 *image,
    SdlWindowHelper sdl_helper
) {
    for (int i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; i++) {
        int x = i % SCREEN_WIDTH;
        int y = i / SCREEN_WIDTH;
        sdl_helper.putPixel(x, y, image[i]);
    }
    sdl_helper.render();
}

// Generates a list of starting positions for the camera and fills the array
/*
void generateCameraStartPositions(
    vec4 *camera_start_positions,
    float *camera_start_yaws
) {
    for (int i = 0; i < num_iterations; i++) {
        int min = -1;
        int max = 1;

        float randx = min + ((float) rand() / (float) RAND_MAX) * (max - min);
        float randy = min + ((float) rand() / (float) RAND_MAX) * (max - min);
        float randz = min + ((float) rand() / (float) RAND_MAX) * (max - min);

        min = 0;
        max = 2 * (float) M_PI;

        float rand_yaw = min + ((float) rand() / (float) RAND_MAX) * (max - min);

        camera_start_positions[i] = vec4(randx, randy, randz, 1.0f);
        camera_start_yaws[i] = rand_yaw;
    }
}
 */

// Initialises the random states for each thread with the same seed
__global__
void render_init(
    curandState * rand_state,
    int seed
) {
    // Assign a thread to each pixel (x, y)
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Calculate the pixel index in the linearised array
    unsigned int pixel_index = (SCREEN_HEIGHT - y - 1) * SCREEN_WIDTH + x;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1927, pixel_index, 0, &rand_state[pixel_index]);
    //curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

// Bulk of the rendering is controlled here
__global__
void render_kernel(
    vec3 *colours,
    vec3 *surface_normals,
    vec3 *albedos,
    float *depths,
    vec3 *colour_variances,
    vec3  *surface_normal_variances,
    vec3 *albedo_variances,
    float *depth_variances,
    Camera camera,
    Triangle *triangles,
    int num_tris,
    Sphere *spheres,
    int num_spheres,
    curandState *rand_state,
    bool is_reference_image,
    Light light,
    LightSphere ls
) {
    int num_samples = is_reference_image ? D_REF_SAMPLES_PER_PIXEL : D_NOISY_SAMPLES_PER_PIXEL;

    // Assign a cuda thread to each pixel (x,y)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // The index of the pixel we are working on when the 2x2 array is linearised
    unsigned int pixel_index = (SCREEN_HEIGHT - y - 1) * SCREEN_WIDTH + x;

    // Flip the y coordinate
    y = SCREEN_HEIGHT - y;

    /*
    // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
    vec4 dir(
            (float) x - SCREEN_WIDTH / 2.0f,
            (float) y - SCREEN_HEIGHT / 2.0f,
            camera.focal_length_,
            1
    );

    // Create a ray for the given pixel
    Ray ray(camera.position_, dir);
    ray.rotateRay(camera.yaw_);

    vec3 colour = ray.raytrace(triangles, num_tris, spheres, num_spheres, light, ls);

    colours[pixel_index] = colour;
     */

    //////////////////////////////////////////////////////////

    vec3 colour_accum = vec3(0.0f);
    vec3 surface_normal_accum = vec3(0.0f);
    vec3 albedo_accum = vec3(0.0f);
    float depth_accum = 0.0f;

    vec3 colour_square_accum = vec3(0.0f);
    vec3 surface_normal_square_accum = vec3(0.0f);
    vec3 albedo_square_accum = vec3(0.0f);
    float depth_square_accum = 0.0f;

    for (int i = 0; i < num_samples; i++) {

        // Give the sample a random direction to give an aliasing effect
        float rand_x = curand_uniform(&rand_state[pixel_index]) - 0.5f;
        float rand_y = curand_uniform(&rand_state[pixel_index]) - 0.5f;

        // Change the ray's direction to work for the current pixel (pixel space -> Camera space)
        vec4 dir(
            (float) (x + rand_x) - SCREEN_WIDTH / 2.0f,
            (float) (y + rand_y) - SCREEN_HEIGHT / 2.0f,
            camera.focal_length_,
            1
        );

        // Create a ray for the given pixel
        Ray ray(camera.position_, dir);
        ray.rotateRay(camera.yaw_);

        vec3 first_sn;
        vec3 first_albedo;
        float first_depth;

        vec3 colour = tracePathIterative(
            ray,
            triangles,
            num_tris,
            spheres,
            num_spheres,
            rand_state,
            NUM_BOUNCES,
            first_sn,
            first_albedo,
            first_depth
        );

        colour_accum += colour;
        surface_normal_accum += first_sn;
        albedo_accum += first_albedo;
        depth_accum += first_depth;

        colour_square_accum += colour * colour;
        surface_normal_square_accum += first_sn * first_sn;
        albedo_square_accum += first_albedo * first_albedo;
        depth_square_accum += first_depth * first_depth;

    }
    colours[pixel_index] = colour_accum / (float) num_samples;
    surface_normals[pixel_index] = surface_normal_accum / (float) num_samples;
    albedos[pixel_index] = albedo_accum / (float) num_samples;
    depths[pixel_index] = depth_accum / (float) num_samples;
    if (depths[pixel_index] < 0) {
        depths[pixel_index] = 0;
    } else if (depths[pixel_index] > 10) {
        printf("%d\n", depths[pixel_index]);
    }

    vec3 colour_var = colour_square_accum / (float) num_samples - colours[pixel_index] * colours[pixel_index];
    vec3 surface_normal_var = surface_normal_square_accum / (float) num_samples -
                              surface_normals[pixel_index] * surface_normals[pixel_index];
    vec3 albedo_var = albedo_square_accum / (float) num_samples - albedos[pixel_index] * albedos[pixel_index];
    float depth_var = depth_square_accum / (float) num_samples - depths[pixel_index] * depths[pixel_index];

    colour_variances[pixel_index] = colour_var;
    surface_normal_variances[pixel_index] = surface_normal_var;
    albedo_variances[pixel_index] = albedo_var;
    depth_variances[pixel_index] = depth_var;
}

void update(Camera &camera, Light &light) {
    static int t = SDL_GetTicks();
    /* Compute frame time */
    int t2 = SDL_GetTicks();
    float dt = float(t2 - t);
    t = t2;

    std::cout << "Render time: " << dt << "ms." << std::endl;

    /* Update variables*/

    const Uint8 *keystate = SDL_GetKeyboardState(NULL);

    if (keystate[SDL_SCANCODE_UP]) {
        camera.moveForwards(0.1);
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
        camera.moveBackwards(0.1);
    }
    if (keystate[SDL_SCANCODE_LEFT]) {
        camera.rotateLeft(0.1);
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
        camera.rotateRight(0.1);
    }
    /*if (keystate[SDL_SCANCODE_A]) {
    }
    if (keystate[SDL_SCANCODE_D]) {
        light.translateRight(0.1);
    }
    if (keystate[SDL_SCANCODE_Q]) {
        light.translateUp(0.1);
    }
    if (keystate[SDL_SCANCODE_E]) {
        light.translateDown(0.1);
    }
    if (keystate[SDL_SCANCODE_W]) {
        light.translateForwards(0.1);
    }
    if (keystate[SDL_SCANCODE_S]) {
        light.translateBackwards(0.1);
    }*/
}

void loadShapes(Triangle *triangles, Sphere *spheres) {
    float cornell_length = 555;            // Length of Cornell Box side.

    vec4 A(cornell_length, 0, 0, 1);
    vec4 B(0, 0, 0, 1);
    vec4 C(cornell_length, 0, cornell_length, 1);
    vec4 D(0, 0, cornell_length, 1);

    vec4 E(cornell_length, cornell_length, 0, 1);
    vec4 F(0, cornell_length, 0, 1);
    vec4 G(cornell_length, cornell_length, cornell_length, 1);
    vec4 H(0, cornell_length, cornell_length, 1);

    // Counter to track triangles
    int curr_tris = 0;
    int curr_spheres = 0;

    // Triangles now take a material as an argument rather than a colour
    // Floor:
    Triangle floor_triangle_1 = Triangle(C, B, A, m_sol_base3);
    //triangles.push_back(floor_triangle_1);
    triangles[curr_tris] = floor_triangle_1;
    curr_tris++;

    Triangle floor_triangle_2 = Triangle(C, D, B, m_sol_base3);
    //triangles.push_back(floor_triangle_2);
    triangles[curr_tris] = floor_triangle_2;
    curr_tris++;

    // Left wall
    Triangle left_wall_1 = Triangle(A, E, C, m_red);
    //triangles.push_back(left_wall_1);
    triangles[curr_tris] = left_wall_1;
    curr_tris++;

    Triangle left_wall_2 = Triangle(C, E, G, m_red);
    //triangles.push_back(left_wall_2);
    triangles[curr_tris] = left_wall_2;
    curr_tris++;

    // Right wall
    Triangle right_wall_1 = Triangle(F, B, D, m_green);
    //triangles.push_back(right_wall_1);
    triangles[curr_tris] = right_wall_1;
    curr_tris++;

    Triangle right_wall_2 = Triangle(H, F, D, m_green);
    //triangles.push_back(right_wall_2);
    triangles[curr_tris] = right_wall_2;
    curr_tris++;

    // Ceiling
    Triangle ceiling_1 = Triangle(E, F, G, m_sol_base3);
    //triangles.push_back(ceiling_1);
    triangles[curr_tris] = ceiling_1;
    curr_tris++;

    Triangle ceiling_2 = Triangle(F, H, G, m_sol_base3);
    //triangles.push_back(ceiling_2);
    triangles[curr_tris] = ceiling_2;
    curr_tris++;


    // Back wall
    Triangle back_wall_1 = Triangle(G, D, C, m_sol_base3);
    //triangles.push_back(back_wall_1);
    triangles[curr_tris] = back_wall_1;
    curr_tris++;

    Triangle back_wall_2 = Triangle(G, H, D, m_sol_base3);
    //triangles.push_back(back_wall_2);
    triangles[curr_tris] = back_wall_2;
    curr_tris++;

    // ----- LIGHTS ----- //
    //float divisor = 1.5f;
    //float diff = cornell_length - (cornell_length / divisor);
    //vec4 new_e = vec4(E.x - diff, E.y - 1, E.z + diff, 1.0f);
    //vec4 new_f = vec4(F.x + diff, F.y - 1, F.z + diff, 1.0f);
    //vec4 new_g = vec4(G.x - diff, G.y - 1, G.z - diff, 1.0f);
    //vec4 new_h = vec4(H.x + diff, H.y - 1, H.z - diff, 1.0f);

    //Triangle light_1 = Triangle(new_e, new_f, new_g, m_light);
    //triangles.push_back(ceiling_1);
    //triangles[curr_tris] = light_1;
    //curr_tris++;

    //Triangle light_2 = Triangle(new_f, new_h, new_g, m_light);
    //triangles.push_back(ceiling_2);
    //triangles[curr_tris] = light_2;
    //curr_tris++;

    // ---------------------------------------------------------------------------
    // Short block

    A = vec4(240, 0, 234, 1);  //+120 in z -50 in x
    B = vec4(80, 0, 185, 1);
    C = vec4(190, 0, 392, 1);
    D = vec4(32, 0, 345, 1);

    E = vec4(240, 165, 234, 1);
    F = vec4(80, 165, 185, 1);
    G = vec4(190, 165, 392, 1);
    H = vec4(32, 165, 345, 1);

    // Front
    //triangles.push_back(Triangle(E, B, A, m_sol_base3));
    triangles[curr_tris] = Triangle(E, B, A, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(E, F, B, m_sol_base3));
    triangles[curr_tris] = Triangle(E, F, B, m_sol_base3);
    curr_tris++;

    // Front
    //triangles.push_back(Triangle(F, D, B, m_sol_base3));
    triangles[curr_tris] = Triangle(F, D, B, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(F, H, D, m_sol_base3));
    triangles[curr_tris] = Triangle(F, H, D, m_sol_base3);
    curr_tris++;

    // BACK
    //triangles.push_back(Triangle(H, C, D, m_sol_base3));
    triangles[curr_tris] = Triangle(H, C, D, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(H, G, C, m_sol_base3));
    triangles[curr_tris] = Triangle(H, G, C, m_sol_base3);
    curr_tris++;

    // LEFT
    //triangles.push_back(Triangle(G, E, C, m_sol_base3));
    triangles[curr_tris] = Triangle(G, E, C, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(E, A, C, m_sol_base3));
    triangles[curr_tris] = Triangle(E, A, C, m_sol_base3);
    curr_tris++;

    // TOP
    //triangles.push_back(Triangle(G, F, E, m_sol_base3));
    triangles[curr_tris] = Triangle(G, F, E, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(G, H, F, m_sol_base3));
    triangles[curr_tris] = Triangle(G, H, F, m_sol_base3);
    curr_tris++;

    // ---------------------------------------------------------------------------
    // Tall block

    A = vec4(443, 0, 247, 1);
    B = vec4(285, 0, 296, 1);
    C = vec4(492, 0, 406, 1);
    D = vec4(334, 0, 456, 1);

    E = vec4(443, 330, 247, 1);
    F = vec4(285, 330, 296, 1);
    G = vec4(492, 330, 406, 1);
    H = vec4(334, 330, 456, 1);

    // Front

    //triangles.push_back(Triangle(E, B, A, m_sol_base3));
    triangles[curr_tris] = Triangle(E, B, A, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(E, F, B, m_sol_base3));
    triangles[curr_tris] = Triangle(E, F, B, m_sol_base3);
    curr_tris++;

    // Front
    //triangles.push_back(Triangle(F, D, B, m_sol_base3));
    triangles[curr_tris] = Triangle(F, D, B, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(F, H, D, m_sol_base3));
    triangles[curr_tris] = Triangle(F, H, D, m_sol_base3);
    curr_tris++;

    // BACK
    //triangles.push_back(Triangle(H, C, D, m_sol_base3));
    triangles[curr_tris] = Triangle(H, C, D, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(H, G, C, m_sol_base3));
    triangles[curr_tris] = Triangle(H, G, C, m_sol_base3);
    curr_tris++;

    // LEFT
    //triangles.push_back(Triangle(G, E, C, m_sol_base3));
    triangles[curr_tris] = Triangle(G, E, C, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(E, A, C, m_sol_base3));
    triangles[curr_tris] = Triangle(E, A, C, m_sol_base3);
    curr_tris++;

    // TOP
    //triangles.push_back(Triangle(G, F, E, m_sol_base3));
    triangles[curr_tris] = Triangle(G, F, E, m_sol_base3);
    curr_tris++;
    //triangles.push_back(Triangle(G, H, F, m_sol_base3));
    triangles[curr_tris] = Triangle(G, H, F, m_sol_base3);
    curr_tris++;

    // ---------------------------------------------------------------------------
    // Sphere

    //Sphere for the light
    //spheres[curr_spheres] = Sphere(vec4(0, -1, 0, 1), 0.2, m_light);
    //curr_spheres++;
    //spheres[curr_spheres] = Sphere(vec4(-0.4, 0.8, -0.5, 1), 0.2, m_sol_magenta);
    //curr_spheres++;


    // ----------------------------------------------
    // Scale to the volume [-1,1]^3

    for (size_t i = 0; i < curr_tris; ++i) {
        triangles[i].v0_ = (triangles[i].v0_ * (2 / cornell_length));
        triangles[i].v1_ = (triangles[i].v1_ * (2 / cornell_length));
        triangles[i].v2_ = (triangles[i].v2_ * (2 / cornell_length));

        triangles[i].v0_ = (triangles[i].v0_ - vec4(1, 1, 1, 1));
        triangles[i].v1_ = (triangles[i].v1_ - vec4(1, 1, 1, 1));
        triangles[i].v2_ = (triangles[i].v2_ - vec4(1, 1, 1, 1));

        vec4 new_v0 = triangles[i].v0_;
        new_v0.x *= -1;
        new_v0.y *= -1;
        new_v0.w = 1.0;
        triangles[i].v0_ = (new_v0);

        vec4 new_v1 = triangles[i].v1_;
        new_v1.x *= -1;
        new_v1.y *= -1;
        new_v1.w = 1.0;
        triangles[i].v1_ = (new_v1);

        vec4 new_v2 = triangles[i].v2_;
        new_v2.x *= -1;
        new_v2.y *= -1;
        new_v2.w = 1.0;
        triangles[i].v2_ = (new_v2);

        triangles[i].computeAndSetNormal();
    }
}
