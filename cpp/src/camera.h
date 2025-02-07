#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

using glm::vec3;
using glm::vec4;
using glm::mat4;

// This class enables the user to view the room and look around by moving the
// camers
class Camera {

    public:
        vec4 position_;
        mat4 rotation_matrix_;
        float yaw_;

        Camera(vec4 position);

        void rotateLeft(float yaw);
        void rotateRight(float yaw);
        void moveForwards(float distance);
        void moveBackwards(float distance);

    private:
        mat4 lookAt(vec3 from, vec3 to);
};

#endif
