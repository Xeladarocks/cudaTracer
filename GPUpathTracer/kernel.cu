
#include "kernel.h"

#define FOV 180

#define SPHERE 1
#define TRIANGLE 2
#define PLANE 3

//__device__ inputPointers pointers;

inline __device__ void getXYZCoords(int& x, int& y, int& z) {
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z + blockDim.z * threadIdx.z;
}

__global__ void cudaRender(inputPointers inpointers, int imgw, int imgh, float currTime) {
	int x, y, z;
	getXYZCoords(x, y, z);

	if (x < imgw && y < imgh) {
		glm::vec3 col(0);
		for (int s = 0; s < inpointers.scene.samples; s++) {
			float randx = (inpointers.random_buffer[(y * imgw + x * (s+1)) % (imgw * imgh)]);
			float randy = (inpointers.random_buffer[(y * imgw + x * (s+1)) % (imgw * imgh)]);
			glm::vec3 direction(
				((((float)x + randx) / (float)imgw) - 0.5f) * (float)(FOV / 180),
				((((float)y + randy) / (float)imgh) - 0.5f) * (float)(FOV / 180) * ((float)imgh / (float)imgw),
				1.0f);
			//direction = inpointers.scene.camera.rotation.rollMat * direction;
			direction = inpointers.scene.camera.rotation.pitchMat * direction;
			direction = inpointers.scene.camera.rotation.yawMat * direction;
			Ray ray = Ray({ inpointers.scene.camera.position, direction });
			int init_depth = 1;
			const unsigned int idx = (y * imgw + x) * (3 + s) % (imgw*imgh);
			col = col + trace(ray, init_depth, idx, inpointers);
			//col += glm::vec3(0, 255, 0);
		}
		col = col * (1.0f/(float)inpointers.scene.samples);
		const unsigned int p_idx = (y * imgw + x) * 4;
		inpointers.image[p_idx] = col.x;
		inpointers.image[p_idx + 1] = col.y;
		inpointers.image[p_idx + 2] = col.z;
		/*inpointers.image[p_idx] = cuda_random_buffer[(y * imgw + x) * 3] * 255;
		inpointers.image[p_idx + 1] = cuda_random_buffer[(y * imgw + x) * 3 + 1] * 255;
		inpointers.image[p_idx + 2] = cuda_random_buffer[(y * imgw + x) * 3 + 2] * 255;*/
	}
}

inline __device__ glm::vec3 trace(Ray &ray, int &depth, const int &idx, inputPointers &inpointers) {
	Intersection intersect = castRay(ray, inpointers);
	if (!intersect.hit)
		return getSkyboxAt(ray.direction, inpointers.scene.skybox);
	Material& material({});
	glm::vec3 ndir(0, -1, 0); // new direction for next ray
	glm::vec3 normal(0, 1, 0);
	glm::vec3 &intersectPoint = intersect.collisionPoint;

	switch (intersect.type) {
		case SPHERE:
			normal = calculateSphereNormal(intersect.sphere, intersectPoint);
			material = intersect.sphere.material;
			break;
		case TRIANGLE:
			normal = calculateTriangleNormal(intersect.triangle, intersectPoint);
			material = intersect.triangle.material;
			break;
		case PLANE:
			normal = intersect.plane.normal;
			material = intersect.plane.material;
			break;
	}
	if (material.diffuse > 0.0f) {
		ndir = diffuse(ray, normal, random_in_unit_sphere(inpointers.random_buffer, idx));
	}
	else if (material.reflectivity > 0.0f) {
		ndir = reflect(ray, normal);
	}
	if (depth <= inpointers.scene.depth && material.emissive <= 0) {
		depth++;
		return glm::clamp(trace(Ray({ intersectPoint + ndir * (float)EPSILON, ndir }), depth, idx, inpointers) * (material.color / 255.0f * inpointers.scene.camera.lightloss), 0.0f, 255.0f); // light loss
	} else
		return material.color * material.emissive;
}

inline __device__ Intersection castRay(Ray &ray, inputPointers &inpointers) {
	float closestDist = -1;
	glm::vec3 nextPos(-1, -1, -1);
	int type = {};

	Sphere& fsphere = {};
	for (int i = 0; i < inpointers.scene.numSpheres; i++) {
		float currDist;
		Sphere &sphere = inpointers.scene.spheres[i];
		bool hit = intersectsSphere(ray, sphere, currDist);
		if (hit && (currDist < closestDist || closestDist < 0)) {
			closestDist = currDist;
			nextPos = ray.position + currDist * ray.direction;
			fsphere = sphere;
			type = SPHERE;
		}
	}
	Triangle& ftriangle = {};
	for (int i = 0; i < inpointers.scene.numTriangles; i++) {
		float currDist;
		Triangle& triangle = inpointers.scene.triangles[i];
		bool hit = intersectsTriangle(ray, triangle, currDist);
		if (hit && (currDist < closestDist || closestDist < 0)) {
			closestDist = currDist;
			nextPos = ray.position + currDist * ray.direction;
			ftriangle = triangle;
			type = TRIANGLE;
		}
	}
	Plane& fplane = {};
	for (int i = 0; i < inpointers.scene.numPlanes; i++) {
		float currDist;
		Plane& plane = inpointers.scene.planes[i];
		bool hit = intersectsPlane(ray, plane, currDist);
		if (hit && (currDist < closestDist || closestDist < 0)) {
			closestDist = currDist;
			nextPos = ray.position + currDist * ray.direction;
			fplane = plane;
			type = PLANE;
		}
	}
	return Intersection({ closestDist > -1, closestDist, nextPos, type, fsphere, ftriangle, fplane});
}