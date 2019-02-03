#include "shape.h"

Shape::Shape(Material material) {
    this->material_ = material;
}

Material Shape::get_material() {
    return this->material_;
}
