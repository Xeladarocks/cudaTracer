
// OpenGL
#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif
#include <GL/glew.h> // GLEW should be included before GLFW
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// CUDA
#include "curand.h"
#include "curand_kernel.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include "helper_cuda_gl.h"

// Project
#include "util.h"

// Extra
#include <ctime>

inline __device__ void getXYZCoords(int& x, int& y, int& z);
__global__ void cudaRender(inputPointers inpointers, int imgw, int imgh, float currTime);
inline __device__ glm::vec3 trace(Ray &ray, int &depth, const int &idx, inputPointers &inpointers);
inline __device__ Intersection castRay(Ray &ray, inputPointers &inpointers);
