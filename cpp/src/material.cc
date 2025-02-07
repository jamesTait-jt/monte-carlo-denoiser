#include "material.h"

Material::Material() {

}

// The constructor simply sets all of the member variables
Material::Material(
    vec3 diffuse_light_component,
    vec3 specular_light_component,
    vec3 emitted_light_component
){
    this->diffuse_light_component_ = diffuse_light_component; 
    this->specular_light_component_ = specular_light_component; 
    this->emitted_light_component_ = emitted_light_component; 
}
