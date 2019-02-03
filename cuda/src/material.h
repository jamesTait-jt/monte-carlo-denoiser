#ifndef MATERIAL_H
#define MATERIAL_H

#include <glm/glm.hpp>

using glm::vec3;

// This class should be used to add materials to polygons. This class determines
// how the polygons will interact with light rays, and thus how they appear
// after being rendered.
class Material {
    public:
        Material();
        Material(
            vec3 diffuse_light_component,
            vec3 specular_light_component,
            vec3 emitted_light_component
        );

        vec3 get_diffuse_light_component();
        vec3 get_specular_light_component();
        vec3 get_emitted_light_component();

        //void set_diffuse_light_component(vec3 diffuse_light_component);
        //void set_specular_light_component(vec3 specular_light_component);
        //void set_emitted_light_component(vec3 emitted_light_component);

    private:
        // Materials are made up of three different types of light. Specular,
        // diffuse, and emitted.
        vec3 diffuse_light_component_;
        vec3 specular_light_component_;
        vec3 emitted_light_component_;
};

#endif
