#include <iostream>

#include <gtest/gtest.h>

// Need this define to allow string_cast to be used
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"

#include "../src/triangle.h"
#include "../src/ray.h"

float floating_point_err = 0.000001;

TEST(TriangleNormalTest, PositiveCoords) {
    vec4 v0(1, 1, 1, 1);
    vec4 v1(1, 2, 1, 1);
    vec4 v2(2, 1, 1, 1);
    Material mat;
    Triangle tri(v0, v1, v2, mat);

    vec3 edge1 = vec3(tri.getV1() - tri.getV0());
    vec3 edge2 = vec3(tri.getV1() - tri.getV2());
    vec3 edge3 = vec3(tri.getV2() - tri.getV0());
    vec3 normal = vec3(tri.getNormal());

    ASSERT_TRUE(glm::dot(edge1, normal) < floating_point_err);
    ASSERT_TRUE(glm::dot(edge2, normal) < floating_point_err);
    ASSERT_TRUE(glm::dot(edge3, normal) < floating_point_err);
}

TEST(TriangleNormalTest, BigPositiveCoords) {
    vec4 v0 = vec4(1430, 109, 10, 1);
    vec4 v1 = vec4(140, 2626, 320, 1);
    vec4 v2 = vec4(210, 150, 4880, 1);
    Material mat;
    Triangle tri(v0, v1, v2, mat);

    vec3 edge1 = vec3(tri.getV1() - tri.getV0());
    vec3 edge2 = vec3(tri.getV1() - tri.getV2());
    vec3 edge3 = vec3(tri.getV2() - tri.getV0());
    vec3 normal = vec3(tri.getNormal());

    ASSERT_TRUE(glm::dot(edge1, normal) < floating_point_err);
    ASSERT_TRUE(glm::dot(edge2, normal) < floating_point_err);
    ASSERT_TRUE(glm::dot(edge3, normal) < floating_point_err);
}

TEST(RayTest, TestNormalised) {
    vec4 start(1, 1, 1, 1);
    vec4 direction(1, 1, 1, 1);
    Ray ray(start, direction);
    vec3 direction3 = vec3(ray.get_direction());
    ASSERT_TRUE(1 - glm::length(direction3) < floating_point_err);
    ASSERT_EQ(1, ray.get_direction()[3]);
}

TEST(RayTest, TestNormalisedNegativeDirection) {
    vec4 start(1, 1, 1, 1);
    vec4 direction(-1, -1, -1, 1);
    Ray ray(start, direction);
    vec3 direction3 = vec3(ray.get_direction());
    ASSERT_TRUE(1 - glm::length(direction3) < floating_point_err);
    ASSERT_EQ(1, ray.get_direction()[3]);
}

TEST(RayTest, TestNormalisedBigDirection) {
    vec4 start(1, 1, 1, 1);
    vec4 direction(12992, 12455, 139, 1);
    Ray ray(start, direction);
    vec3 direction3 = vec3(ray.get_direction());
    ASSERT_TRUE(1 - glm::length(direction3) < floating_point_err);
    ASSERT_EQ(1, ray.get_direction()[3]);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
