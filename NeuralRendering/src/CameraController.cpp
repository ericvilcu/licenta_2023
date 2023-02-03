#include "CameraController.hpp"
#if WIN32
#include <Windows.h>
#endif
#include <iostream>
#include "HeaderThatSupressesWarnings.h"
#include <SDL.h>
#include <SDL_rect.h>
#include <SDL_opengl.h>
#include <gl/gl.h>
#include <gl/glu.h>
#include "HeaderThatReenablesWarnings.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>


void CameraController::processMovements()
{
	float3 dir{ 0,0,0 };
	float3 rot{ 0,0,0 };
	SDL_GL_MakeCurrent((SDL_Window*)renderer.window, renderer.context);
	int nk;
	const Uint8* keys = SDL_GetKeyboardState(&nk);
	if (keys[SDL_SCANCODE_A])dir.x -= 0.2f;
	if (keys[SDL_SCANCODE_D])dir.x += 0.2f;
	if (keys[SDL_SCANCODE_LSHIFT]||keys[SDL_SCANCODE_RSHIFT])dir.y += 0.2f;
	if (keys[SDL_SCANCODE_SPACE])dir.y -= 0.2f;
	if (keys[SDL_SCANCODE_W])dir.z -= 0.2f;
	if (keys[SDL_SCANCODE_S])dir.z += 0.2f;
	if (keys[SDL_SCANCODE_Q])rot.z -= 0.02f;
	if (keys[SDL_SCANCODE_E])rot.z += 0.02f;
	//todo: make holding down the button not flip back and forth every frame.
	if (keys[SDL_SCANCODE_F])camera->flip_x();
	if (keys[SDL_SCANCODE_N])camera->use_neural = !camera->use_neural;

	if (keys[SDL_SCANCODE_V]) {
		dir.x *= 5; dir.y *= 5; dir.z *= 5;
		if (keys[SDL_SCANCODE_C]) {
			dir.x *= 5; dir.y *= 5; dir.z *= 5;
		}
	} else {
		if (keys[SDL_SCANCODE_C]) {
			dir.x /= 10; dir.y /= 10; dir.z /= 10;
		}
	}

	if (SDL_GetMouseFocus() == renderer.window) {
		int x, y;
		int state = SDL_GetMouseState(&x, &y);
		if (!(state & SDL_BUTTON_LEFT)) {
			mouse_trustable = false;
		}
		else {
			if (mouse_trustable) {
				rot.x = (float)(x - lastx) / 1e2f;
				rot.y = (float)(y - lasty) / 1e2f;
				if (rot.x != 0.0f || rot.y != 0.0f)
					SDL_WarpMouseInWindow((SDL_Window*)renderer.window, lastx, lasty);
			}
			else {
				lastx = x;
				lasty = y;
			}
			mouse_trustable = true;
		}
	}
	else mouse_trustable = false;

	SDL_PumpEvents();
	camera->move(dir);
	camera->rotate_clampy(rot,PI/2.1f);
}
