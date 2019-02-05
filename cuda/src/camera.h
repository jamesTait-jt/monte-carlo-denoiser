#ifndef CAMERA_H
#define CAMERA_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <glm/glm.hpp>

using glm::vec3;
using glm::vec4;
using glm::mat4;

// This class enables the user to view the room and look around by moving the
// camers
class Camera {

    public:
        vec4 position_;
        float yaw_;

        CUDA_HOSTDEV Camera(vec4 position);

        void rotateLeft(float yaw);
        void rotateRight(float yaw);
        void moveForwards(float distance);
        void moveBackwards(float distance);

        vec4 get_position();
        mat4 get_rotation_matrix();
        float get_yaw();

    private:
        mat4 rotation_matrix_;

        mat4 lookAt(vec3 from, vec3 to);
};

#endif
