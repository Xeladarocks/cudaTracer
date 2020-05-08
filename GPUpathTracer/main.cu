
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


#define PERFORMANCE_DEBUG
#define PI 3.14159
const unsigned int WIDTH = 800, HEIGHT = 600;

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

/*** Scene buffers and definitions ***/
Camera camera({ glm::vec3(0, 3, 0), Rotation({PI, PI, PI, glm::mat3(0), glm::mat3(0), glm::mat3(0)}), Controls({false, false, false, false, false, false}) });

const int NUM_SPHERES = 3;
Sphere scene_spheres[NUM_SPHERES];
size_t size_scene_spheres;
void* cuda_scene_spheres; // sphere buffer
/*** ----- ------- --- ----------- ***/

/** CUDA **/
size_t size_random_data = num_texels * 3 * sizeof(float);
float random_buffer[num_texels * 3];
float* cuda_random_buffer; // random number buffer

void* cuda_dev_render_buffer; // stores output

struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result

float cudaTime;
cudaEvent_t start, stop;
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
	keyboardfunct(window, key, scancode, action, mods, camera);
}
void mouseFunc(GLFWwindow* window, double xpos, double ypos) {
	mouseFunct(window, xpos, ypos, camera);
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
	glfwSwapInterval(0);

	glfwSetKeyCallback(window, keyboardfunc);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // unlimited mouse motion
	glfwSetCursorPosCallback(window, mouseFunc);
	return true;
}
void createSpheres() {
	size_scene_spheres = sizeof(Sphere) * NUM_SPHERES;

	scene_spheres[0] = Sphere({ glm::vec3(0, 1, 10), 2, Material({glm::vec3(255, 255, 0), 1.0f, 0.0f, 0.0f}) });
	scene_spheres[1] = Sphere({ glm::vec3(2, 5, 10), 1.5, Material({glm::vec3(0, 125, 255), 0.0f, 0.0f, 3.0f}) });
	scene_spheres[2] = Sphere({ glm::vec3(0, -1001, 10), 1000, Material({glm::vec3(255, 255, 255), 1.0f, 0.0f, 0.0f}) });

	checkCudaErrors(cudaDeviceSynchronize());
}
void createRandoms() {
	checkCudaErrors(cudaMalloc(&cuda_random_buffer, size_random_data)); // Allocate CUDA memory for buffer
	for (int i = 0; i < num_texels * 3; i++) {
		random_buffer[i] = (float)randomDouble() * 2.0f - 1.0f;
	}
	checkCudaErrors(cudaDeviceSynchronize());
}
void prepScene() {
	createSpheres();
	createRandoms();

	updateMatrxs(camera.rotation);
}
void initCUDABuffers() {
	size_t myStackSize = 8192;
	cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);

	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output

	checkCudaErrors(cudaMalloc(&cuda_scene_spheres, size_scene_spheres)); // Allocate CUDA memory for scene data
}
void updateObjects(std::chrono::duration<double> deltaTime, std::chrono::duration<double> duration) {

	scene_spheres[1].position.z = 10 + 6 * std::sinf(duration.count());
	scene_spheres[1].position.x = 6 * std::cosf(duration.count());
	scene_spheres[1].position.y = 5 + 4 * std::sinf(2*duration.count());


	// update buffers
	checkCudaErrors(cudaMemcpy(cuda_scene_spheres, scene_spheres, size_scene_spheres, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cuda_random_buffer, random_buffer, size_random_data, cudaMemcpyHostToDevice));
}
void cudaEndTimer(float time, cudaEvent_t start, cudaEvent_t stop) {
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

	printf("Time to generate:  %3.1f ms \n", time);
}
void generateCUDAImage(std::chrono::duration<double> totalTime, std::chrono::duration<double> deltaTime) {
	// calculate grid size
	dim3 block(10, 10, 1);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1); // 2D grid, every thread will compute a pixel

	updateObjects(deltaTime, totalTime);
	updateCamera(camera);
	const float* yawMatVals = (const float*)glm::value_ptr(camera.rotation.yawMat);

	sceneInfo info{ (Camera)camera, (Sphere*)cuda_scene_spheres, NUM_SPHERES,    (float)totalTime.count() };
	inputPointers pointers{ (unsigned int*)cuda_dev_render_buffer, info };

	#ifdef PERFORMANCE_DEBUG
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	#endif
	cudaRender << < grid, block >> > (pointers, (float*)cuda_random_buffer, WIDTH, HEIGHT, (float)totalTime.count());
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
	generateCUDAImage(duration, deltaTime);
	glfwPollEvents();
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	// Swap the screen buffers
	glfwSwapBuffers(window);
}


int main(int argc, char* argv[]) {
	initGLFW();
	initGL();

	printGLFWInfo(window);
	printGlewInfo();
	printGLInfo();

	prepScene();

	findCudaGLDevice(argc, (const char**)argv);
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
	int frameNum = 0;
	// Some computation here


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

		display(totalTime, currTime - lastTime);
		//std::cout << abs(sin(totalTime.count() / pow(10, 7)))*255 << "\n";
		std::chrono::duration<double> elapsed_seconds = currTime - lastMeasureTime;
		frameNum++;
		if (elapsed_seconds.count() >= 1.0) {
			// show fps every  second
			std::cout << "fps: " << (frameNum / elapsed_seconds.count()) << "\n";
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
	glBindVertexArray(0); // unbind VAO


	glfwDestroyWindow(window);
	glfwTerminate();

	system("pause"); // don't close console
	exit(EXIT_SUCCESS);

}