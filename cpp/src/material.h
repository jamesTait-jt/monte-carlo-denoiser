#ifndef MATERIAL_H
#define MATERIAL_H

#include <glm/glm.hpp>

using glm::vec3;

// This class should be used to add materials to polygons. This class determines
// how the polygons will interact with light rays, and thus how they appear
// after being rendered.
class Material {
    public:
        // Materials are made up of three different types of light. Specular,
        // diffuse, and emitted.
        vec3 diffuse_light_component_;
        vec3 specular_light_component_;
        vec3 emitted_light_component_;

        Material();
        Material(
            vec3 diffuse_light_component,
            vec3 specular_light_component,
            vec3 emitted_light_component
        );
};

#endif
