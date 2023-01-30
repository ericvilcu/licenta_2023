#pragma once
#include <string>
#include <memory>
#include <map>
#include "cuda.h"
typedef void* SDL_WindowPointer;// Window = SDL_CreateWindow("OpenGL Test", 0, 0, WinWidth, WinHeight, WindowFlags);
class Renderer
{
public:
	struct View{
		unsigned int textureId=0;
		int width=0,height=0,top=0,left=0;
		//percent left,right,up,down. use to create multiple views in a single window.
		float lp=0,rp=0,up=0,dp=0;
    	//Flag discerns wether the width/height/up/left needs automatic update based on lp,rp,up,dp and width/height of the viewport
		bool needs_update=true,resize_on_window_resize=false;
		//currently unused.
		int priority=0;
	};
	enum ViewTypeEnum:int{
		MAIN_VIEW = 0,
		POINTS_VIEW = 1,
		TRAIN_VIEW_1 = 2,
		TRAIN_VIEW_2 = 3,
	};
	typedef int ViewType;
public://private:
	std::map<int,std::unique_ptr<View>> views;

	SDL_WindowPointer window;
	void* context;
	unsigned int texture;

	unsigned int program, buffer;
	int window_width, window_height;
public:
	Renderer(Renderer&) = delete;
	Renderer(Renderer&&) = delete;
	Renderer&operator=(Renderer&) = delete;
	Renderer&operator=(Renderer&&) = delete;
	Renderer() = delete;
	Renderer(const std::string& window_title);
	/**
	 * @brief Creates or overwrites a new view with ID vt
	 * @param vt id. if a view with this id already exists, it will be overwritten.
	 * @param lp what percentage of the width will be to the left of the view.
	 * @param rp what percentage of the width will be to the left of the right edge of the view.
	 * @param up what percentage of the height will be above the view.
	 * @param dp what percentage of the height will be above the bottom of the view.
	 * @param priority higher priority gets drawn on lower priority. < 0 essentially means it's disabled.
	 */
	void createView(ViewType vt, float lp, float rp, float up, float dp, int priority, bool resize_on_window_resize = true);
	void removeView(ViewType vt);
	const View& getView(ViewType vt) const {
		return *views.at(vt);
	};
	View& getView(ViewType vt) {
		return *views.at(vt);
	};
	void uploadRGBA8(ViewType vt,void*data,int width, int height);
	void ensureWH(int width, int height);
	//It's funny that, as bytecode, these two functions are practically the same.
	void getWH(int& width, int& height)const;
	void getWH(int* width, int* height)const;
	void update();
	~Renderer();
	struct RenderSetupError {
		std::string data;
		RenderSetupError(std::string data) :data{ data }{}
	};
	bool shouldClose();
};

