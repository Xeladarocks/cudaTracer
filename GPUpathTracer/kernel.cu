
#include "kernel.h"


#define FOV 180

__device__ inputPointers pointers;
__device__ float** random_data;

inline __device__ void getXYZCoords(int& x, int& y, int& z) {
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z + blockDim.z * threadIdx.z;
}

__global__ void cudaRender(inputPointers inpointers, float* cuda_random_buffer, int imgw, int imgh, float currTime) {
	int x, y, z;
	getXYZCoords(x, y, z);
	
	pointers = inpointers;

	glm::vec3 position = pointers.scene.camera.position;
	glm::vec3 direction = glm::vec3(
		((((float)x + 0.5) / (float)imgw) - 0.5f) * (float)(FOV / 180),
		((((float)y + 0.5) / (float)imgh) - 0.5f) * (float)(FOV / 180) * ((float)imgh / (float)imgw),
		1.0f);
	direction = direction * pointers.scene.camera.rotation.rollMat;
	direction = direction * pointers.scene.camera.rotation.pitchMat;
	direction = direction * pointers.scene.camera.rotation.yawMat;
	Ray ray = Ray({ position, direction });
	glm::vec3 col = trace(ray, 1, (y * imgw + x) * 3, cuda_random_buffer);
	//glm::vec3 col(0, 255, 0);

	int firstPos = (y * imgw + x) * 4;
	pointers.image[firstPos] = col.x;
	pointers.image[firstPos + 1] = col.y;
	pointers.image[firstPos + 2] = col.z;
}

inline __device__ glm::vec3 trace(Ray &ray, int depth, int idx, float* cuda_random_buffer) {
	Intersection intersect = castRay(ray);
	if (!intersect.hit) {
		return glm::vec3(0);
	}
	glm::vec3 normal = calculateSphereNormal(intersect.sphere, intersect.collisionPoint);
	glm::vec3 ndir(0, -1, 0); // new direction for next ray
	if (intersect.sphere.material.diffuse > 0.0f) {
		ndir = diffuse(ray, normal, random_in_unit_sphere(cuda_random_buffer, idx));
	} else if(intersect.sphere.material.reflectivity > 0.0f) {
		ndir = reflect(ray, normal);
	}
	if (depth <= 1 && intersect.sphere.material.emissive <= 0) {
		return glm::clamp(trace(Ray({ intersect.collisionPoint + ndir * 0.001f, ndir }), depth + 1, idx, cuda_random_buffer) * (intersect.sphere.material.color / 255.0f * 0.8f), 0.0f, 255.0f); // light loss
	} else {
		return intersect.sphere.material.color * intersect.sphere.material.emissive;
	}
}

inline __device__ Intersection castRay(Ray &ray) {
	float closestDist = -1;
	glm::vec3 nextPos(-1, -1, -1);
	Sphere fsphere;
	for (int i = 0; i < pointers.scene.numSpheres; i++) {
		float currDist;
		Sphere sphere = pointers.scene.spheres[i];
		bool hit = intersectsSphere(ray, sphere, currDist);
		if (hit && (currDist < closestDist || closestDist < 0)) {
			closestDist = currDist;
			nextPos = ray.position + currDist * ray.direction;
			fsphere = sphere;
		}
	}
	return Intersection({ closestDist > -1, closestDist, nextPos, fsphere});
}