
// Project
#include "kernel.h"
#include "cutil.hpp"
#include "input.hpp"

// Additional
#include <string>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// Project Helpers
#include "GLSLProgram.h"
#include "GLSLShader.h"
#include "gl_tools.h"
#include "glfw_tools.h"

#include <glm/gtc/type_ptr.hpp>

/*** Constants ***/
//#define PERFORMANCE_DEBUG
#define PI 3.14159
constexpr auto TIME_TO_ENHANCEMENT = 15;
constexpr auto TARGET_SAMPLE_COUNT = 50;
constexpr auto TARGET_BOUNCE_DEPTH = 1;
constexpr auto REALTIME_SAMPLE_COUNT = 1;
constexpr auto REALTIME_BOUNCE_DEPTH = 1;
constexpr auto LOCK_CONTROLS = false;

std::string scene_file_name = "cornellBox.txt";

// calculate grid size
dim3 block(32, 32, 1);
const int WIDTH = nearestMultiple(400, block.x), HEIGHT = nearestMultiple(400, block.y);
dim3 grid(WIDTH / block.x, HEIGHT / block.y); // 2D grid, every thread will compute a pixel
/*** --------- ***/

// GLFW
GLFWwindow* window;

// OpenGL
GLuint VBO, VAO, EBO;
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

const int num_texels = WIDTH * HEIGHT;
int num_values = num_texels * 4;
int size_tex_data = sizeof(GLuint) * num_values;

/*** Scene buffers ***/
const int NUM_SPHERES = 2;
Sphere scene_spheres[NUM_SPHERES];
size_t size_scene_spheres;
void* cuda_scene_spheres; // sphere buffer

const int NUM_TRIANGLES = 20;
Triangle scene_triangles[NUM_TRIANGLES];
size_t size_scene_triangles;
void* cuda_scene_triangles; // triangle buffer

const int NUM_PLANES = 1;
Plane scene_planes[NUM_PLANES];
size_t size_scene_planes;
void* cuda_scene_planes; // plane buffer

size_t size_random_data = num_texels * 3 * sizeof(float);
float* random_buffer = new float[num_texels * 3];
float* cuda_random_buffer; // random number buffer

size_t size_scene_info;
void* cuda_scene_info; // scene info buffer

void* cuda_dev_render_buffer; // stores output

struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result
/*** ----- ------- --- ----------- ***/


/** CUDA definitions **/
float cudaTime;
cudaEvent_t start, stop;

bool updateFrame = true;
bool moving = true;
bool updateNextFrame = true;
bool rendering = false;
bool rendered = false;
int samples = REALTIME_SAMPLE_COUNT;
int depth = REALTIME_BOUNCE_DEPTH;
bool updateMeasured = false;

Camera camera({ glm::vec3(0, 10, -25), 1.0f, Rotation({0, 0, 0, glm::mat3(0), glm::mat3(0), glm::mat3(0)}), Controls({false, false, false, false, false, false}) });
Skybox skybox({ glm::vec3(0, 1, 0), glm::vec3(63, 178, 232), glm::vec3(225, 244, 252), glm::vec3(225, 244, 252), false, glm::vec3(0), 1 });

sceneInfo info{ samples, depth, (Camera)camera, (Skybox)skybox, (Sphere*)cuda_scene_spheres, NUM_SPHERES, (Triangle*)cuda_scene_triangles, NUM_TRIANGLES, (Plane*)cuda_scene_planes, NUM_PLANES,    0.0f };
/** ---- **/


static const char* glsl_drawtex_vertshader_src =
"#version 430 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec2 texCoord;\n"
"\n"
"out vec2 ourTexCoord;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0f);\n"
"	ourTexCoord = texCoord;\n"
"}\n";

static const char* glsl_drawtex_fragshader_src =
"#version 430 core\n"
"uniform usampler2D tex;\n"
"in vec2 ourTexCoord;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"   	vec4 c = texture(tex, ourTexCoord);\n"
"   	color = c / 255.0;\n"
"}\n";

// QUAD GEOMETRY
GLfloat vertices[] = {
	// Positions          // Texture Coords
	 1.0f,  1.0f, 0.5f,   1.0f, 1.0f,  // Top Right
	 1.0f, -1.0f, 0.5f,   1.0f, 0.0f,  // Bottom Right
	-1.0f, -1.0f, 0.5f,   0.0f, 0.0f,  // Bottom Left
	-1.0f,  1.0f, 0.5f,   0.0f, 1.0f,  // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {
	0, 1, 3,
	1, 2, 3
};

// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y) {
	// create an OpenGL texture
	glGenTextures(1, gl_tex); // generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI_EXT, size_x, size_y, 0, GL_RGB_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	// Register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	SDK_CHECK_ERROR_GL();
}
void initGLBuffers() {
	// create texture that will receive the result of cuda kernel
	createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, WIDTH, HEIGHT);
	// create shader program
	drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
	drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
	shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
	shdrawtex.compile();
	SDK_CHECK_ERROR_GL();
}
void initCUDABuffers() {
	size_t myStackSize = 8192;
	cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);

	// Allocate CUDA memory for color output
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));

	// Allocate CUDA memory for random data
	checkCudaErrors(cudaMalloc(&cuda_random_buffer, size_random_data));
	checkCudaErrors(cudaMemcpy(cuda_random_buffer, random_buffer, size_random_data, cudaMemcpyHostToDevice));

	/* Allocate CUDA memory for scene data */
	// Spheres
	checkCudaErrors(cudaMalloc(&cuda_scene_spheres, size_scene_spheres));
	checkCudaErrors(cudaMemcpy(cuda_scene_spheres, scene_spheres, size_scene_spheres, cudaMemcpyHostToDevice));
	// Triangles
	checkCudaErrors(cudaMalloc(&cuda_scene_triangles, size_scene_triangles));
	checkCudaErrors(cudaMemcpy(cuda_scene_triangles, scene_triangles, size_scene_triangles, cudaMemcpyHostToDevice));
	// Planes
	checkCudaErrors(cudaMalloc(&cuda_scene_planes, size_scene_planes));
	checkCudaErrors(cudaMemcpy(cuda_scene_planes, scene_planes, size_scene_planes, cudaMemcpyHostToDevice));
}
bool initGL() {
	glewExperimental = GL_TRUE; // need this to enforce core profile
	GLenum err = glewInit();
	glGetError(); // parse first error
	if (err != GLEW_OK) {// Problem: glewInit failed, something is seriously wrong.
		printf("glewInit failed: %s /n", glewGetErrorString(err));
		exit(1);
	}
	glViewport(0, 0, WIDTH, HEIGHT); // viewport for x,y to normalized device coordinates transformation
	SDK_CHECK_ERROR_GL();
	return true;
}
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (!LOCK_CONTROLS && rendering == false) {
		keyboardfunct(window, key, scancode, action, mods, camera);
		if (key != GLFW_KEY_C)updateMeasured = true;
	}
	if(key == GLFW_KEY_C)render_target_settings();
}
void mouseFunc(GLFWwindow* window, double xpos, double ypos) {
	if (!LOCK_CONTROLS && rendering == false) {
		mouseFunct(window, xpos, ypos, camera);
		updateMeasured = true;
	}
}
bool initGLFW() {
	if (!glfwInit()) exit(EXIT_FAILURE);
	// These hints switch the OpenGL profile to core
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "Raytracer", NULL, NULL);
	if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }
	GLFWmonitor* monitor = getBestMonitor(window);
	centerWindow(window, monitor);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glfwSetKeyCallback(window, keyboardfunc);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // unlimited mouse motion
	glfwSetCursorPosCallback(window, mouseFunc);
	return true;
}
void createObjects() {
	loadSceneData(scene_file_name, scene_spheres, scene_triangles, scene_planes);

	size_scene_spheres = sizeof(Sphere) * NUM_SPHERES;
	size_scene_triangles = sizeof(Triangle) * NUM_TRIANGLES;
	size_scene_planes = sizeof(Plane) * NUM_PLANES;

	checkCudaErrors(cudaDeviceSynchronize());
}
void updateObjects(std::chrono::duration<double> deltaTime, std::chrono::duration<double> duration) {
	scene_spheres[1].position.z = 100 * std::sinf(duration.count());
	scene_spheres[1].position.x = 100 * std::cosf(duration.count());
	scene_spheres[1].position.y = 100 + 5 * std::sinf(2 * duration.count());

	// update scene buffers
	//checkCudaErrors(cudaMemcpy(cuda_scene_spheres, scene_spheres, size_scene_spheres, cudaMemcpyHostToDevice));
}
void createRandoms() {
	checkCudaErrors(cudaMalloc(&cuda_random_buffer, size_random_data)); // Allocate CUDA memory for buffer
	for (int i = 0; i < num_texels * 3; i++) {
		random_buffer[i] = (float)randomDouble();
	}
	checkCudaErrors(cudaDeviceSynchronize());
}
void prepScene() {
	createObjects();
	createRandoms();
	updateMatrxs(camera.rotation);
}
void generateCUDAImage(std::chrono::duration<double> totalTime, std::chrono::duration<double> deltaTime) {

	updateObjects(deltaTime, totalTime);
	updateCamera(camera);
	const float* yawMatVals = (const float*)glm::value_ptr(camera.rotation.yawMat);

	info = { samples, depth, (Camera)camera, (Skybox)skybox, (Sphere*)cuda_scene_spheres, NUM_SPHERES, (Triangle*)cuda_scene_triangles, NUM_TRIANGLES, (Plane*)cuda_scene_planes, NUM_PLANES,    (float)totalTime.count() };
	inputPointers pointers{ (unsigned int*)cuda_dev_render_buffer, info, (float*)cuda_random_buffer };

	#ifdef PERFORMANCE_DEBUG
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	#endif
	cudaRender << < grid, block >> > (pointers, WIDTH, HEIGHT, (float)totalTime.count());
	checkCudaErrors(cudaDeviceSynchronize());
	#ifdef PERFORMANCE_DEBUG
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&cudaTime, start, stop));
	printf("Time to generate CUDA:  %.2f ms\n", cudaTime);
	#endif

	cudaArray* texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

	cudaDeviceSynchronize();
}

void display(std::chrono::duration<double> duration, std::chrono::duration<double> deltaTime) {
	glClear(GL_COLOR_BUFFER_BIT);
	if(updateFrame)
		generateCUDAImage(duration, deltaTime);
	if (!updateNextFrame)updateFrame = false;
	glfwPollEvents();
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	// Swap the screen buffers
	glfwSwapBuffers(window);
}
void updateMeasuredValue() { updateMeasured = true; }
void render_target_settings() {
	if (!updateNextFrame)return;
	samples = TARGET_SAMPLE_COUNT;
	depth = TARGET_BOUNCE_DEPTH;
	updateNextFrame = false;
	moving = false;
	updateMeasured = false;
	printf("Target Samples: %i\n", samples);
	printf("Target Depth: %i\n", depth);
}
void accumulativeUpdates(std::chrono::system_clock::time_point currTime, std::chrono::system_clock::time_point &lastMeasuredSampleTime) {
	if (updateMeasured == true) {
		lastMeasuredSampleTime = currTime;
		updateMeasured = false;
		updateNextFrame = true;
		updateFrame = true;
		moving = true;
		rendering = false;
		rendered = false;
		samples = REALTIME_SAMPLE_COUNT;
		depth = REALTIME_BOUNCE_DEPTH;
	}
	if (moving == false && updateFrame == true) {
		render_target_settings();
	}
	if (moving == true && std::chrono::duration_cast<std::chrono::seconds>(currTime - lastMeasuredSampleTime).count() >= TIME_TO_ENHANCEMENT) {
		lastMeasuredSampleTime = currTime;
		moving = false;
	}
	/*if (increasingSamples == true && std::chrono::duration_cast<std::chrono::seconds>(currTime - lastMeasuredSampleTime).count() >= 0.2 && samples < TARGET_SAMPLE_COUNT && GRADUAL_ACCUMULATION) {
		// slowly increasing sample and depth
		float sampleIncrease = (float)((float)std::chrono::duration_cast<std::chrono::seconds>(currTime - lastMeasuredSampleTime).count() / 0.2f);
		samples += (sampleIncrease+samples >= TARGET_SAMPLE_COUNT)?TARGET_SAMPLE_COUNT-samples:sampleIncrease;
		float depthIncrease = (float)((float)std::chrono::duration_cast<std::chrono::seconds>(currTime - lastMeasuredSampleTime).count() / 0.2f);
		depth += (depthIncrease + depth >= TARGET_BOUNCE_DEPTH) ? TARGET_BOUNCE_DEPTH - depth : depthIncrease;

		printf("Samples: %i\n", samples);
		printf("Depth: %i\n", depth);
	}*/
}
int main(int argc, char* argv[]) {
	initGLFW();
	initGL();

	printGLFWInfo(window);
	printGlewInfo();
	printGLInfo();

	prepScene();

	//pick the device with highest Gflops / s
	cudaDeviceProp deviceProp;
	int devID = gpuGetMaxGflopsDeviceId();
	checkCudaErrors(cudaSetDevice(devID));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
	printf("CUDA GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	initGLBuffers();
	initCUDABuffers();

	// Generate buffers
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Buffer setup
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO); // all next calls wil use this VAO (descriptor for VBO)

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Texture attribute (2 floats)
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
	glBindVertexArray(0);

	// Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
	// A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
	// This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.
	auto firstTime = std::chrono::system_clock::now();
	auto lastTime = firstTime;
	auto lastMeasureTime = firstTime;
	auto targetTime = firstTime;
	int frameNum = 0;
	
	auto lastMeasuredSampleTime = firstTime;

	glBindVertexArray(VAO); // binding VAO automatically binds EBO
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

	shdrawtex.use(); // we gonna use this compiled GLSL program
	glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);
	SDK_CHECK_ERROR_GL();


	while (!glfwWindowShouldClose(window)) {
		auto currTime = std::chrono::system_clock::now();
		auto totalTime = currTime - firstTime;

		if (updateNextFrame == false && rendered == false) {
			rendering = true;
			printf("Starting target render...\n");
			targetTime = currTime;
		}
		accumulativeUpdates(currTime, lastMeasuredSampleTime);
		display(totalTime, currTime - lastTime);
		if (rendering == true) {
			rendering = false;
			rendered = true;
			auto endRender = std::chrono::system_clock::now();
			auto timeTaken = endRender - targetTime;
			printf("Target render completed in: %02i:%02i:%02is\n", std::chrono::duration_cast<std::chrono::hours>(timeTaken).count(), std::chrono::duration_cast<std::chrono::minutes>(timeTaken).count(), std::chrono::duration_cast<std::chrono::seconds>(timeTaken).count());
		}
		else if (rendered == false) {
			//std::cout << abs(sin(totalTime.count() / pow(10, 7)))*255 << "\n";
			std::chrono::duration<double> elapsed_seconds = currTime - lastMeasureTime;
			frameNum++;
			if (elapsed_seconds.count() >= 1.0) {
				// show fps every  second
				printf("  ----- fps: %f -----  \n", (float)(frameNum / elapsed_seconds.count()));
				frameNum = 0;
				lastMeasureTime = currTime;
			}
			#ifdef PERFORMANCE_DEBUG
			printf("Time to show frame:     %.2f ms\n", (float)std::chrono::duration_cast<std::chrono::milliseconds>(currTime - lastTime).count() - cudaTime);
			printf("Time to generate frame: %.2f ms\n", (float)std::chrono::duration_cast<std::chrono::milliseconds>(currTime - lastTime).count());
			printf("----------------------------------\n");
			#endif
			lastTime = currTime;
		}
	}
	glBindVertexArray(0); // unbind VAO


	glfwDestroyWindow(window);
	glfwTerminate();

	system("pause"); // don't close console
	exit(EXIT_SUCCESS);

}