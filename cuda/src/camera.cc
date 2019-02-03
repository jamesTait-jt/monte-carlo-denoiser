#include "camera.h"

#include <iostream>

Camera::Camera(vec4 position) {
    this->position_ = position;
    this->yaw_ = 0;
    this->rotation_matrix_ = mat4(1.0);
}

void Camera::rotateLeft(float yaw) {
    this->yaw_ = (this->yaw_ - yaw);
    mat4 new_rotation_matrix = this->rotation_matrix_;
    new_rotation_matrix[0] = vec4(cos(-yaw), 0, sin(-yaw), 0);
    new_rotation_matrix[2] = vec4(-sin(-yaw), 0, cos(-yaw), 0);
    this->rotation_matrix_ = new_rotation_matrix;
    this->position_ = this->rotation_matrix_ * this->position_;
}

void Camera::rotateRight(float yaw) {
    this->yaw_ = (this->yaw_ + yaw);
    mat4 new_rotation_matrix = this->rotation_matrix_;
    new_rotation_matrix[0] = vec4(cos(yaw), 0, sin(yaw), 0);
    new_rotation_matrix[2] = vec4(-sin(yaw), 0, cos(yaw), 0);
    this->rotation_matrix_ = new_rotation_matrix;
    this->position_ = this->rotation_matrix_ * this->position_;
}

void Camera::moveForwards(float distance) {
    vec3 new_camera_pos(
        this->position_[0] - distance * sin(this->yaw_),
        this->position_[1],
        this->position_[2] + distance * cos(this->yaw_)
    );
    mat4 cam_to_world = lookAt(new_camera_pos, vec3(0, 0, 0));
    this->position_ = cam_to_world * vec4(0, 0, 0, 1);
}

void Camera::moveBackwards(float distance) {
    vec3 new_camera_pos(
        this->position_[0] + distance * sin(this->yaw_),
        this->position_[1],
        this->position_[2] - distance * cos(this->yaw_)
    );
    mat4 cam_to_world = lookAt(new_camera_pos, vec3(0, 0, 0));
    this->position_ = cam_to_world * vec4(0, 0, 0, 1);
}

mat4 Camera::lookAt(vec3 from, vec3 to) {
    vec3 forward = normalize(from - to);
    vec3 temp(0, 1, 0);
    vec3 right = cross(normalize(temp), forward);
    vec3 up = cross(forward, right);

    vec4 forward4(forward.x, forward.y, forward.z, 0);
    vec4 right4(right.x, right.y, right.z, 0);
    vec4 up4(up.x, up.y, up.z, 0);
    vec4 from4(from.x, from.y, from.z, 1);

    mat4 cam_to_world(right4, up4, forward4, from4);

    return cam_to_world;
}

vec4 Camera::get_position() {
    return this->position_;
}

mat4 Camera::get_rotation_matrix() {
    return this->rotation_matrix_;
}

float Camera::get_yaw() {
    return this->yaw_;
}
