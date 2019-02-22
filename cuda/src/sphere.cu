#include "sphere.h"
#include "util.h"

Sphere::Sphere(){}

Sphere::Sphere(vec4 centre, float radius, Material material) {
    centre_ = centre;
    radius_ = radius;
    material_ = material;
}

__device__
bool Sphere::solveQuadratic(
    const float & a,
    const float & b,
    const float & c,
    float & x0,
    float & x1
) {
    float discr = b * b - 4 * a * c;
    if (discr < 0) {
        return false;
    }
    else if (discr == 0) {
        x0 = x1 = (float) - 0.5 * b / a;
    }
    else {
        float q = (b > 0) ?
                  -0.5f * (b + sqrtf(discr)) :
                  -0.5f * (b - sqrtf(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) {
        swap(x0, x1);
    }
    return true;
}
