#ifndef LIGHTSPHERE_H
#define LIGHTSPHERE_H

#include <glm/glm.hpp>
#include <vector>

#include "light.h"

// This class is a sphere containing multiple point lights. Using this method
// gives us nice soft shadows rather than sharply moving from light to dark
class LightSphere {

    public:
        LightSphere(vec4 centre, float radius, int num_lights, float intensity, vec3 colour);
        vec3 directLight(Intersection intersection, std::vector<Shape *> shapes);

        std::vector<Light> get_point_lights();
        vec4 get_centre();
        float get_radius();
        float get_intensity();
        vec3 get_colour();
    
    private:
        std::vector<Light> point_lights_;
        vec4 centre_;
        float radius_;
        float intensity_;
        vec3 colour_;

        std::vector<Light> sphereSample(int numLights);
        bool containedInSphere(vec4 p);

};

#endif
