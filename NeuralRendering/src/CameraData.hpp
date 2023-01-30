#pragma once
#include <math.h>
#include "dtype4x4.cuh"

//.cu files break when you use typedef, but I still want decent intellisense&cloring
#ifdef __INTELLISENSE__
typedef int CameraType_t;
#else
#define CameraType_t int
#endif

enum CameraType:int{
	GL,
	PINHOLE_PROJECTION,
};

//Interface for main camera data.
class CameraDataItf {
public:
	virtual CameraType_t type() const = 0;
	virtual ~CameraDataItf() {};
	virtual int get_width() const = 0;
	virtual int get_height() const= 0;
	virtual std::unique_ptr<CameraDataItf> scaleTo(int w, int h) const = 0;
};
//Template for Partial Classes
struct PartialCameraDataTemplate {
	int w; int h;
	PartialCameraDataTemplate() { h = w = 0; };
	PartialCameraDataTemplate(int w, int h) :w{ w }, h{ h } {};
	/*
	To be used in tamplates, implement methods:
	__hdfi__ float4 mapToWorldCoords(float4 in) const;
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float4 in) const;
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float4 in) const{
		return mapToScreenCoords(mapToWorldCoords());
	};
	__hdfi__ float3 DirectionForPixel(int2 uv)
	*/
};
struct ScreenCoordsWithDepth {
	int2 coords;
	float depth;
	bool valid;
	__hdfi__ ScreenCoordsWithDepth(int2 coords, float depth, bool valid = true) {
		this->coords = coords;
		this->depth = depth;
		this->valid = valid;
	}
	static __hdfi__ ScreenCoordsWithDepth invalid() {
		return ScreenCoordsWithDepth(make_int2(0,0),0.0,false);
	}
};
/*Here for reference, delete later.
bool __device__ __forceinline__ mapScreenCooords(float4 coords, int2& out, const int w, const int h, const PartialGLCameraData camera, float* depth = NULL) {
	float4 position = camera.transform * coords;
	position = camera.perspective * position;
	if (position.z < camera.near_clip || position.z > camera.far_clip) return false;
	position.x /= position.w;
	position.y /= position.w;
	if (depth != NULL) *depth = position.z;
	int dx = (position.x / 2 + 0.5) * w;
	int dy = (position.y / 2 + 0.5) * h;
	if (dx < 0 || dy < 0 || dx >= w || dy >= h)return false;
	out.x = dx;
	out.y = dy;
	return true;
}*/


/*
* When a struct is sent to a kernel, all of its components are sent as well. This is used to truncate the CameraData struct to only contain the essentials
*/
struct PartialGLCameraData : public PartialCameraDataTemplate {
public:
	bool use_neural = false;
	float4x4 transform = float4x4();
	float4x4 perspective = float4x4();
	float near_clip = 0, far_clip = 1e20f;
	PartialGLCameraData() :PartialCameraDataTemplate{-1, -1} {}
	PartialGLCameraData(float near_clip, float far_clip, int w, int h) : PartialCameraDataTemplate{ w,h }, transform{}, near_clip{ near_clip }, far_clip{ far_clip }
		{}
	//These should be implemented for any camera type, or else the inline thing in kernels may break.
	__hdfi__ float4 mapToWorldCoords(float4 coords) const{
		return transform * coords;
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float4 world_coords) const{
		float4 position = perspective * world_coords;
		if (position.z < near_clip || position.z > far_clip) return ScreenCoordsWithDepth::invalid();
		position.x /= position.w;
		position.y /= position.w;
		float depth = position.z;
		int dx = (int)((position.x / 2 + 0.5) * w);
		int dy = (int)((position.y / 2 + 0.5) * h);
		if (dx < 0 || dy < 0 || dx >= w || dy >= h)return ScreenCoordsWithDepth::invalid();
		return ScreenCoordsWithDepth(make_int2(dx,dy),depth);
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float4 coords) const{
		return mapToScreenCoordsFromWorldCoords(mapToWorldCoords(coords));
	};
	__hdfi__ float3 direction_for_pixel(int2 uv) const{
		float3 direction;
		direction.z = 1;
		direction.x = 2 * ((float)uv.x / w - 0.5f);
		direction.y = 2 * ((float)uv.y / h - 0.5f);
		return transform.inverted_direction(perspective.inverted_direction(direction));
	}
};
constexpr float PI = 3.14159265358979323851f;
/*
* Camera data, full struct. Used for freeCam. Just what the standard OpenGL camera is with some extra data to make moving it around easy. The only camera type that adapts to viewport size for now.
*/
struct CameraGLData:CameraDataItf,PartialGLCameraData {
	virtual CameraType_t type() const { return CameraType::GL; };
	virtual int get_width() const { return w; };
	virtual int get_height() const { return h; };
	virtual std::unique_ptr<CameraDataItf> scaleTo(int w, int h) const {
		auto scaled=std::make_unique<CameraGLData>(*this); scaled->w = w; scaled->h = h;
		return scaled;
	};
	float scaleY,fov_x,fov_rad;
	float yaw, pitch, roll;
	bool flipped_x;
	float3 translation;
	CameraGLData() :PartialGLCameraData{},
		scaleY{ 1 }, fov_x{ 0 }, fov_rad{ 0 },
		yaw{ 0 }, pitch{ 0 }, roll{ 0 }, flipped_x{ false },
		translation{ 0,0,0 } {};
	CameraGLData(float scaleY,float fov_rad,float near_clip,float far_clip,float yaw=0,float pitch=0,float roll=0,bool flipped_x=false,float3 translation=make_float3(0,0,0),int w=-1,int h=-1)
		:PartialGLCameraData{ near_clip,far_clip,w,h }, 
		scaleY{ scaleY }, fov_rad{ fov_rad },
		yaw{ yaw }, pitch{ pitch }, roll{ roll }, flipped_x{ flipped_x },
		translation{ translation }
	{ recalculateTangent(); recalculateTransform(); }
	CameraGLData(const CameraGLData& ot) = default;
	void recalculateTangent() {
		fov_x = (float) tan(fov_rad/2);
	}

	void setFov(float rad) {
		fov_rad = rad;
		recalculateTangent();
	}

	void move(float3 dir) {
		float3 transformed_dir = transform.inv_rotated(dir);
		//I think up/down should maybe not go by the same rules.
		translation.x += transformed_dir.x;
		translation.y += transformed_dir.y;
		translation.z += transformed_dir.z;
		recalculateTransform();
	}

	void flip_x() {
		flipped_x =! flipped_x;
	}

	void rotate(float2 dir) {
		yaw = fmodf(yaw + dir.x * (flipped_x ? -1 : 1), PI * 2);
		pitch = fmodf(pitch + dir.y, PI * 2);
		recalculateTransform();
	}

	void rotate_clampy(float3 dir,float max_y) {
		yaw = fmodf(yaw + dir.x * (flipped_x ? -1 : 1), PI * 2);
		roll = fmodf(roll + dir.z, PI * 2);
		pitch += dir.y;
		if (pitch > max_y)pitch = max_y;
		else if (pitch < -max_y)pitch = -max_y;
		recalculateTransform();
	}

	void recalculateTransform() {
		float y = yaw; float p = pitch; float r = roll;
		//somewhat taken from http://msl.cs.uiuc.edu/planning/node102.html
		//with alpha=roll, beta=-yaw, gamma=pitch
		transform[0][0] = (float)(cos(r)*cos(y));
		transform[1][0] = (float)(sin(r)*cos(y));
		transform[2][0] = (float)(sin(y));
		transform[0][1] = (float)(cos(r)*-sin(y)*sin(p)-sin(r)*cos(p));
		transform[1][1] = (float)(sin(r)*-sin(y)*sin(p)+cos(r)*cos(p));
		transform[2][1] = (float)(cos(y)*sin(p));
		transform[0][2] = (float)(cos(r)*-sin(y)*cos(p)+sin(r)*sin(p));
		transform[1][2] = (float)(sin(r)*-sin(y)*cos(p)-cos(r)*sin(p));
		transform[2][2] = (float)(cos(y)*cos(p));
		
		float4 transformed_dir = make_float4(translation.x, translation.y, translation.z, 0);
		transformed_dir = transform * transformed_dir;
		transform[3][0] = transformed_dir.x;
		transform[3][1] = transformed_dir.y;
		transform[3][2] = transformed_dir.z;
		if (flipped_x) {
			for(int i=0;i<4;++i)
				transform[i][0] *= -1;
		}
	}

	void recalculatePerspective(float h, float w) {
		perspective = float4x4{
			1 / fov_x, 0, 0, 0,
			0, 1 / (fov_x * h / w), 0, 0,
			0, 0, 1, 0,//-camera.far_clip / (camera.far_clip - camera.near_clip),-(camera.near_clip * camera.far_clip) / (camera.far_clip - camera.near_clip),
			0, 0, -1, 0,
		};
	}

	PartialGLCameraData prepareForGPU(int height, int width) const {
		//Intentional slicing
		PartialGLCameraData data;
		data.far_clip = far_clip;
		data.near_clip = near_clip;
		data.transform = transform;
		data.w = width;
		data.h = height;
		//Using a full-on perspective matrix is a little overkill.
		data.perspective = float4x4{
			1 / fov_x, 0, 0, 0,
			0, scaleY / (fov_x * height / width), 0, 0,
			0, 0, 1, 0,
			0, 0, -1, 0,
		};
		return data;
	}
	virtual ~CameraGLData() override {};
};


//Pinhole projection camera data.
struct PartialPinholeCameraData :PartialCameraDataTemplate {
	float4x4 transform = float4x4();
	float ppx,ppy,fx,fy;
	float near_clip, far_clip;

	//These should be implemented for any camera type, or else the inline thing in kernels may break.
	__hdfi__ float4 mapToWorldCoords(float4 coords) const {
		return transform * coords;
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float4 world_coords) const {
		if (world_coords.z < near_clip || world_coords.z > far_clip) return ScreenCoordsWithDepth::invalid();
		float inv_depth = 1/ world_coords.z;

		int dx = (int)(ppx - fx * world_coords.x * inv_depth);
		int dy = (int)(ppy - fy * world_coords.y * inv_depth);
		if (dx < 0 || dy < 0 || dx >= w || dy >= h)return ScreenCoordsWithDepth::invalid();
		return ScreenCoordsWithDepth(make_int2(dx, dy), world_coords.z);
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float4 coords) const {
		return mapToScreenCoordsFromWorldCoords(mapToWorldCoords(coords));
	};
	__hdfi__ float3 direction_for_pixel(int2 uv) const {
		float3 direction;
		direction.z = 1;
		direction.x = (ppx - (float)uv.x) / (fx * direction.z);
		direction.y = (ppy - (float)uv.y) / (fy * direction.z);
		return transform.inverted_direction(direction);
	}
	PartialPinholeCameraData() { ppx = ppy = fx = fy = near_clip = far_clip = 0; };
};

struct PinholeCameraData :CameraDataItf, PartialPinholeCameraData {
	virtual CameraType_t type() const { return CameraType::PINHOLE_PROJECTION; };
	virtual int get_width() const { return w; };
	virtual int get_height() const { return h; };
	virtual std::unique_ptr<CameraDataItf> scaleTo(int w, int h) const {
		auto scaled = std::make_unique<PinholeCameraData>(*this); scaled->w = w; scaled->h = h;
		float dx = (1.f * w / this->w);
		float dy = (1.f * h / this->h);
		scaled->fx *= dx; scaled->fy *= dy;
		//scaled->inv_fx = 1 / scaled->fx; scaled->inv_fy = 1 / scaled->fy;
		scaled->ppx*= dx; scaled->ppy*= dy;
		return scaled;
	};
	PartialPinholeCameraData prepareForGPU(int height, int width) const {
		//This is intentional slicing. The only real things sliced off are v-tables and such
		return *this;
	}
	PinholeCameraData() {};
	PinholeCameraData(const PinholeCameraData& ot) = default;
	PinholeCameraData(PinholeCameraData&& ot) = default;
};
