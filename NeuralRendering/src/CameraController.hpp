#pragma once
#include "Renderer.hpp"
#include "CameraData.hpp"

class CameraController {
	Renderer& renderer;
	std::shared_ptr<InteractiveCameraData> camera;
	int lastx = 0, lasty = 0;
	bool mouse_trustable = false;
	//this is awful. should have used SDL_PumpEvents or something, but at least it works.
	enum SWITCHES {
		FLIP, NEURAL, MONO,
		CHANNEL_PLUS, CHANNEL_MINUS,
		SCENE_PLUS, SCENE_MINUS,
		SHOW_MIPS,
		NUM_SWITCHES
	};
	bool pressed_last_frame[NUM_SWITCHES];
	inline bool check_toggle(int switch_type, bool pressed_this_frame)
	{
		bool ret = !pressed_last_frame[switch_type] && pressed_this_frame;
		pressed_last_frame[switch_type] = pressed_this_frame;
		return ret;
	}
public:
	CameraController(Renderer& renderer, std::shared_ptr<InteractiveCameraData> camera) :renderer{ renderer }, camera{ camera } {
	}
	void processMovements();
};
