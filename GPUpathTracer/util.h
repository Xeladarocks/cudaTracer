#pragma once

#define EPSILON 0.1

/** Ray Tracing **/
struct Ray {
    glm::vec3 position;
    glm::vec3 direction;
};
struct Material {
    glm::vec3 color;
    float diffuse;
    float reflectivity;
    float emissive;
};
struct Sphere {
    glm::vec3 position;
    float radius;
    Material material;
};
struct Rotation {
    float pitch;
    float yaw;
    float roll;
    glm::mat3 yawMat;
    glm::mat3 pitchMat;
    glm::mat3 rollMat;
};
struct Controls {
    bool w;
    bool a;
    bool s;
    bool d;
    bool up;
    bool down;
};
struct Camera {
    glm::vec3 position;
    Rotation rotation;
    Controls controls;
};
inline __device__ bool intersectsSphere(Ray &ray, Sphere &sphere, float &t) {
    glm::vec3 oc = ray.position - sphere.position;
    float k1 = glm::dot(ray.direction, ray.direction);
    float k2 = 2 * glm::dot(oc, ray.direction);
    float k3 = glm::dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = k2 * k2 - 4 * k1 * k3;
    if (discriminant < 0) {
        return false;
    }
    float t1 = (-k2 - sqrt(discriminant)) / (2 * k1);
    float t2 = (-k2 + sqrt(discriminant)) / (2 * k1);
    if (t1 > EPSILON) {
        t = t1;
        return true;
    }
    if (t2 > EPSILON) {
        t = t2;
        return true;
    }
    return false;
}
inline __device__ glm::vec3 calculateSphereNormal(Sphere &sphere, glm::vec3 &point) {
    return glm::normalize(point - sphere.position);
}
inline __device__ glm::vec3 random_in_unit_sphere(float* random_data, int idx) {
    glm::vec3 randomDirection = glm::vec3({
        random_data[idx],
        random_data[idx+1],
        random_data[idx+2]
    });
    return randomDirection;
}
inline __device__ glm::vec3 diffuse(Ray ray, glm::vec3 normal, glm::vec3 dir) {
    if (glm::dot(normal, dir) <= 0) {
        return dir * -1.0f;
    }
    return dir;
}
inline __device__ glm::vec3 reflect(Ray ray, glm::vec3 normal) {
    return ray.direction - (2 * glm::dot(ray.direction, glm::normalize(normal)) * glm::normalize(normal));
}

struct Intersection {
    bool hit;
    double distance;
    glm::vec3 collisionPoint;
    Sphere sphere;
};
/** --- ------- **/


/** CUDA **/
struct sceneInfo {
    Camera camera;
    Sphere* spheres;
    int numSpheres;

    float currTime;
};
struct inputPointers {
    unsigned int* image; // texture position
    sceneInfo scene;
};
/** ---- **/


/** Extra **/
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
/** ----- **/

