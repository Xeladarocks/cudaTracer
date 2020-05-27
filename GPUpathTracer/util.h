  #pragma once

#define EPSILON 0.001

/** Ray Tracing **/
struct Ray {
    glm::vec3 &position;
    glm::vec3 &direction;
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
struct Triangle {
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 v3;
    Material material;
};
struct Plane {
    glm::vec3 position;
    glm::vec3 normal;
    Material material;
};
struct Intersection {
    bool hit;
    double distance;
    glm::vec3 collisionPoint;
    int type;
    Sphere& sphere;
    Triangle& triangle;
    Plane& plane;
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
    float lightloss;
    Rotation rotation;
    Controls controls;
};
struct Skybox {
    glm::vec3 up;
    glm::vec3 topColor;
    glm::vec3 sideColor;
    glm::vec3 bottomColor;
    bool override;
    glm::vec3 overrideColor;
    float intensity;
};
inline __device__ bool intersectsSphere(Ray &ray, Sphere &sphere, float &t) {
    glm::vec3 oc = ray.position - sphere.position;
    float k1 = glm::dot(ray.direction, ray.direction);
    float k2 = 2 * glm::dot(oc, ray.direction);
    float k3 = glm::dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = k2 * k2 - 4 * k1 * k3;
    if (discriminant < 0) return false;
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
inline __device__ bool intersectsTriangle(Ray& ray, Triangle& triangle, float& t) {
    glm::vec3 edge1 = triangle.v2 - triangle.v1;
    glm::vec3 edge2 = triangle.v3 - triangle.v1;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) return false;
    float f = 1.0f / a;
    glm::vec3 s = ray.position - triangle.v1;
    float u = f * glm::dot(s, h);
    if (u < 0 || u > 1) return false;
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    if (v < 0 || u + v > 1) return false;
    float t0 = f * glm::dot(edge2, q);
    if (t0 > EPSILON && t0 < 1 / EPSILON) {
        t = t0;
        return true;
    }
    return false;
}
inline __device__ bool intersectsPlane(Ray &ray, Plane &plane, float& t) {
    if (glm::dot(plane.normal, -ray.direction) > 0) {
        t = glm::dot(plane.position - ray.position, plane.normal) / glm::dot(plane.normal, ray.direction);
        return (t >= 0);
    }
    return false;
}

inline __device__ glm::vec3 calculateSphereNormal(Sphere &sphere, glm::vec3 &point) {
    return glm::normalize(point - sphere.position);
}
inline __device__ glm::vec3 calculateTriangleNormal(Triangle &triangle, glm::vec3 &point) {
    glm::vec3 edge1 = triangle.v2 - triangle.v1;
    glm::vec3 edge2 = triangle.v3 - triangle.v1;
    glm::vec3 cross = glm::cross(edge1, edge2);
    return glm::normalize(cross);
}
inline __device__ glm::vec3 random_in_unit_sphere(float* &random_data, int idx) {
    glm::vec3 randomDirection = glm::vec3({random_data[idx]*2-1,random_data[idx+1]*2-1,random_data[idx+2]*2-1});
    return randomDirection;
}
inline __device__ glm::vec3 diffuse(Ray &ray, glm::vec3 &normal, glm::vec3 &dir) {
    if (glm::dot(normal, dir) <= EPSILON) return dir * -1.0f;
    return dir;
}
inline __device__ glm::vec3 reflect(Ray &ray, glm::vec3 &normal) {
    return ray.direction - (2 * glm::dot(ray.direction, glm::normalize(normal)) * glm::normalize(normal));
}
inline __device__ glm::vec3 getSkyboxAt(glm::vec3 &dir, Skybox &skybox) {
    if (skybox.override)return skybox.overrideColor;
    float dot = glm::dot(skybox.up, dir);
    if (dot < 0)
        return glm::mix(skybox.sideColor, skybox.bottomColor, abs(dot)) * skybox.intensity;
    return glm::mix(skybox.sideColor, skybox.topColor, dot) * skybox.intensity;
}
/** --- ------- **/


/** CUDA **/
struct sceneInfo {
    int samples;
    int depth;
    Camera camera;
    Skybox skybox;

    Sphere* spheres;
    int numSpheres;
    Triangle* triangles;
    int numTriangles;
    Plane* planes;
    int numPlanes;

    float currTime;
};
struct inputPointers {
    unsigned int* image; // texture position
    sceneInfo scene;
    float* random_buffer;
};
/** ---- **/


/** Extra **/

/** ----- **/

