#pragma once
#include "Renderer.hpp"
#include "CameraData.hpp"

class CameraController {
	Renderer& renderer;
	CameraGLData& camera;
	int lastx = 0, lasty = 0;
	bool mouse_trustable = false;
public:
	CameraController(Renderer& renderer, CameraGLData& camera) :renderer{ renderer }, camera{ camera } {
	}
	void processMovements();
};
