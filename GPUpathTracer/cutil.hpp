#pragma once
#include "util.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void updateMatrxs(Rotation& rotation) {
    rotation.pitchMat = glm::mat3(
        1, 0, 0,
        0, cos(rotation.pitch), -sin(rotation.pitch),
        0, sin(rotation.pitch), cos(rotation.pitch));
    rotation.yawMat = glm::mat3(
        cos(rotation.yaw), 0, sin(rotation.yaw),
        0, 1, 0,
        -sin(rotation.yaw), 0, cos(rotation.yaw));
    rotation.rollMat = glm::mat3(
        cos(rotation.roll), -sin(rotation.roll), 0,
        sin(rotation.roll), cos(rotation.roll), 0,
        0, 0, 1);
}