#pragma once
#include "util.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <sstream>
#include <iostream>
#include <regex>

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first)return str;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}
void loadSceneData(std::string &scene_file_name, Sphere* spheres, Triangle* triangles, Plane* planes) {
    std::ifstream ifs(scene_file_name);
    std::string line;
    int line_num = 1;
    while (std::getline(ifs, line)) {
        line = trim(line);  // remove whitespace
        if (line.rfind("//", 0) == 0) continue;
        std::istringstream ss(line);
        if (line.rfind("Sphere", 0) == 0) {
            std::regex rx("Sphere\\( vec3\\((\\d*\\.?\\d*), (\\d*\\.?\\d*), (\\d*\\.?\\d*)\\), (\\d*\\.?\\d*), Material\\( vec3\\((\d*\\.?\\d*), (\\d*\\.?\\d*), (\d*\\.?\\d*)\\), (\\d*\\.?\\d*), (\\d*\\.?\\d*), (\\d*\\.?\\d*) \\)");
            std::smatch match;
            if (std::regex_search(line, match, rx)) {
                for (std::size_t i = 1; i < match.size(); ++i) {
                    std::ssub_match sub_match = match[i];
                    std::string num = sub_match.str();
                    std::cout << " submatch " << i << ": " << num << std::endl;
                }
            } else {
                std::cout << "Warning: incorrect Sphere config on line: " << line_num << "\n";
                continue;
            }
        }
        line_num++;
    }
}

#include <string>
#include <random>
#include <memory>

struct xorshift_engine {
    using result_type = uint32_t;
    uint32_t state;
    xorshift_engine() {
        state = 0x1234567;
    }
    xorshift_engine(uint32_t seed) {
        if (seed == 0) seed++;
        state = seed;
    }
    const uint32_t min() {
        return 1;
    }
    const uint32_t max() {
        return 0xffffffff;
    }
    uint32_t operator()() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
static thread_local std::uniform_real_distribution<double> dist(0.0f, 1.0f);
static thread_local std::random_device rd;
static thread_local xorshift_engine eng(rd());
static double randomDouble() {
    return dist(eng);
}
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
const int nearestMultiple(const int numToRound, const int multiple) {
    if (multiple == 0)
        return numToRound;

    int remainder = abs(numToRound) % multiple;
    if (remainder == 0)
        return numToRound;

    if (numToRound < 0)
        return -(abs(numToRound) - remainder);
    else
        return numToRound + multiple - remainder;
}
