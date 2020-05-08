#pragma once

#include <GLFW/glfw3.h>
#include <cstdio>
#include <string>

using namespace std;

void printGLFWInfo(GLFWwindow* w){
	int p = glfwGetWindowAttrib(w, GLFW_OPENGL_PROFILE);
	string version = glfwGetVersionString();
	string opengl_profile = "";
	if(p == GLFW_OPENGL_COMPAT_PROFILE){
		opengl_profile = "OpenGL Compatibility Profile";
	}
	else if (p == GLFW_OPENGL_CORE_PROFILE){
		opengl_profile = "OpenGL Core Profile";
	}
	printf("GLFW: %s \n", version.c_str());
	printf("GLFW: %s %i \n", opengl_profile.c_str(), p);
}

void centerWindow(GLFWwindow* window, GLFWmonitor* monitor)
{
    if (!monitor)
        return;

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (!mode)
        return;

    int monitorX, monitorY;
    glfwGetMonitorPos(monitor, &monitorX, &monitorY);

    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    glfwSetWindowPos(window,
        monitorX + (mode->width - windowWidth) / 2,
        monitorY + (mode->height - windowHeight) / 2);
}

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((b) < (a)) ? (b) : (a))
GLFWmonitor* getBestMonitor(GLFWwindow* window)
{
    int monitorCount;
    GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);

    if (!monitors)
        return NULL;

    int windowX, windowY, windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glfwGetWindowPos(window, &windowX, &windowY);

    GLFWmonitor* bestMonitor = NULL;
    int bestArea = 0;

    for (int i = 0; i < monitorCount; ++i)
    {
        GLFWmonitor* monitor = monitors[i];

        int monitorX, monitorY;
        glfwGetMonitorPos(monitor, &monitorX, &monitorY);

        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (!mode)
            continue;

        int areaMinX = MAX(windowX, monitorX);
        int areaMinY = MAX(windowY, monitorY);

        int areaMaxX = MIN(windowX + windowWidth, monitorX + mode->width);
        int areaMaxY = MIN(windowY + windowHeight, monitorY + mode->height);

        int area = (areaMaxX - areaMinX) * (areaMaxY - areaMinY);

        if (area > bestArea)
        {
            bestArea = area;
            bestMonitor = monitor;
        }
    }

    return bestMonitor;
}