#pragma once

void keyboardfunct(GLFWwindow* window, int key, int scancode, int action, int mods, Camera &camera) {
	bool swtch;
	if (action == GLFW_PRESS) swtch = true;
	else if (action == GLFW_RELEASE) swtch = false;
	else return;

	switch(key){
		case GLFW_KEY_W: camera.controls.w = swtch; break;
		case GLFW_KEY_S: camera.controls.s = swtch; break;
		case GLFW_KEY_A: camera.controls.a = swtch; break;
		case GLFW_KEY_D: camera.controls.d = swtch; break;
		case GLFW_KEY_SPACE: camera.controls.up = swtch; break;
		case GLFW_KEY_LEFT_SHIFT: camera.controls.down = swtch; break;
	}
}

bool firstMouse = true;
double mouseDeltaX;
double mouseDeltaY;

double lastX;
double lastY;
void mouseFunct(GLFWwindow* window, double xpos, double ypos, Camera &camera) {
	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.01;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	camera.rotation.yaw += xoffset;
	camera.rotation.pitch += yoffset;
	updateMatrxs(camera.rotation);
}

void updateCamera(Camera &camera) {
	if (camera.controls.w) {
		glm::vec3 direction = glm::vec3(0, 0, -1) * camera.rotation.yawMat;
		camera.position = camera.position + direction; // * speedModifier * (deltaTime/50)
	}
	if (camera.controls.s) {
		glm::vec3 direction = glm::vec3(0, 0, -1) * camera.rotation.yawMat;
		camera.position = camera.position - direction; // * speedModifier * (deltaTime/50)
	}
	if (camera.controls.a) {
		glm::vec3 direction = glm::vec3(-1, 0, 0) * camera.rotation.yawMat;
		camera.position = camera.position - direction; // *(deltaTime / 50); // * speedModifier
	}
	if (camera.controls.d) {
		glm::vec3 direction = glm::vec3(-1, 0, 0) * camera.rotation.yawMat;
		camera.position = camera.position + direction; // *(deltaTime / 50); // * speedModifier
	}
	if (camera.controls.up) {
		camera.position = camera.position + glm::vec3(0, 1, 0); // * (deltaTime / 50); // * speedModifier
	}
	if (camera.controls.down) {
		camera.position = camera.position - glm::vec3(0, 1, 0); // *(deltaTime / 50); // * speedModifier
	}
}