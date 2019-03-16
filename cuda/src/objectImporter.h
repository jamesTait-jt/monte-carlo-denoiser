#ifndef OBJECT_IMPORTER_H
#define OBJECT_IMPORTER_H

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <unordered_map>

#include "triangle.h"

using glm::vec3;

bool load_scene(
    const char * path,
    std::vector<Triangle> & triangles,
    std::unordered_map<std::string, Material> material_map
);

void build_triangles(
    std::vector<Triangle> & triangles,
    std::vector<vec3> & vertex_indices,
    std::vector<vec3> & temp_vertices,
    std::vector<std::string> materials,
    std::unordered_map<std::string, Material> material_map
);

void split_string(
    std::vector<std::string> & sub_strs,
    std::string search_string,
    std::string delimiter
);

#endif