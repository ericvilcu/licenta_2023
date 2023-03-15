#pragma once
#include <math.h>
#include <memory>
#include "dtype4x4.cuh"

//.cu files break when you use typedef, but I still want decent intellisense&cloring
#ifdef __INTELLISENSE__
typedef int CameraType_t;
#else
#define CameraType_t int
#endif

enum CameraType:int{
	INTERACTIVE = -1,
	PINHOLE_PROJECTION = 0,
	RADIAL = 1,
};

//Interface for main camera data.
class CameraDataItf {
public:
	virtual CameraType_t type() const = 0;
	virtual ~CameraDataItf() {};
	virtual int get_width() const = 0;
	virtual int get_height() const = 0;
	virtual std::unique_ptr<CameraDataItf> scaleTo(int w, int h) const = 0;
	virtual std::unique_ptr<CameraDataItf> subSection(int x1, int y1, int x2, int y2) const = 0;
	virtual std::string serialized(bool text) const = 0;
	static std::unique_ptr<CameraDataItf> from_serial(bool text, std::istream& data);
	static std::unique_ptr<CameraDataItf> from_serial(bool text, const std::string& data);
	static float4x4 transform_from(float yaw, float pitch, float roll, float3 translation = make_float3(0,0,0)) {
		float y = yaw; float p = pitch; float r = roll;
		float4x4 transform; transform.zero();
		//somewhat taken from http://msl.cs.uiuc.edu/planning/node102.html
		//with alpha=roll, beta=-yaw, gamma=pitch
		transform[0][0] = (float)(cos(r) * cos(y));
		transform[1][0] = (float)(sin(r) * cos(y));
		transform[2][0] = (float)(sin(y));
		transform[0][1] = (float)(cos(r) * -sin(y) * sin(p) - sin(r) * cos(p));
		transform[1][1] = (float)(sin(r) * -sin(y) * sin(p) + cos(r) * cos(p));
		transform[2][1] = (float)(cos(y) * sin(p));
		transform[0][2] = (float)(cos(r) * -sin(y) * cos(p) + sin(r) * sin(p));
		transform[1][2] = (float)(sin(r) * -sin(y) * cos(p) - cos(r) * sin(p));
		transform[2][2] = (float)(cos(y) * cos(p));

		float4 transformed_dir = make_float4(translation.x, translation.y, translation.z, 0);
		transformed_dir = transform * transformed_dir;
		transform[3][0] = transformed_dir.x;
		transform[3][1] = transformed_dir.y;
		transform[3][2] = transformed_dir.z;
		transform[3][3] = 1;

		return transform;
	}
};
//Template for Partial Classes
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
		return ScreenCoordsWithDepth(make_int2(0, 0), 0.0, false);
	}
};

template <typename selfclass>
struct PartialCameraDataTemplate {
	int w; int h;
	float4x4 transform = float4x4();
	float near_clip = 0, far_clip = 1e20f;
	void implementations_exist() {
		//makes sure these all required functions exist, yet should be optimized away.
		me().mapToScreenCoords3(me().direction_for_pixel(make_float2(0, 0)));
	}
	PartialCameraDataTemplate() { h = w = 0; implementations_exist(); };
	PartialCameraDataTemplate(int w, int h) :w{ w }, h{ h } {
		implementations_exist();
	};
	/*
	To be used in tamplates, implement methods:
	__hdfi__ float4 mapToWorldCoords(float4 in) const;
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float4 in) const;
	__hdfi__ float3 direction_for_pixel(float2 uv)
	*/
	__hdfi__ const selfclass& me() const {
		return (const selfclass&)*this;
	}
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords3(float3 coords) const {
		return me().mapToScreenCoords(make_float4(coords.x, coords.y, coords.z, 1));
	}
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float4 coords) const {
		return me().mapToScreenCoordsFromWorldCoords(me().mapToWorldCoords(coords));
	}
	//tl;dr high numbers=bad, if first is <3 it is somewhat ok, if second < 0.3 it is very ok, if second > 2 it is VERY bad.
	static std::pair<float,float> check_internal_consistency(const selfclass& cam) {
		float mx = 0, sm = 0;
		const int pad=5;
		for (int i = pad; i < cam.w - pad; ++i) {
			for (int j = pad; j < cam.h - pad; ++j) {
				const float4x4& transform = cam.transform;
				float3 dir = cam.direction_for_pixel(make_float2(i+0.5f, j+0.5f));
				std::cerr << dir.x << ' ' << dir.y << ' ' << dir.z << '\n';
				float depth = 1;
				float3 ct = transform.translation();
				std::cerr << ct.x << ' ' << ct.y << ' ' << ct.z << '\n';
				ScreenCoordsWithDepth out = cam.mapToScreenCoords(make_float4(dir.x * depth + ct.x, dir.y * depth + ct.y, dir.z * depth + ct.z, 1));
				float dt = abs(out.coords.x - i) + abs(out.coords.y - j);
				std::cerr << dt << ' ' << out.coords.x << ' ' << out.coords.y << ' ' << out.depth << '\n';
				if (out.valid) {
					sm += dt;
					if (dt > mx)mx = dt;
				}
				else return std::make_pair(-1.0f,-1.0f);
			}
		}
		return std::make_pair(mx, sm/((cam.w-pad*2)*(cam.h- pad * 2)));
	}
};
/*Here for reference, delete later.
bool __device__ __forceinline__ mapScreenCooords(float4 coords, int2& out, const int w, const int h, const PartialInteractiveCameraData camera, float* depth = NULL) {
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
struct PartialInteractiveCameraData : public PartialCameraDataTemplate<PartialInteractiveCameraData> {
public:
	bool use_neural = false;
	float4x4 perspective = float4x4();
	PartialInteractiveCameraData() :PartialCameraDataTemplate{-1, -1} {}
	PartialInteractiveCameraData(float near_clip, float far_clip, int w, int h) : PartialCameraDataTemplate{ w,h }
	{ this->near_clip = near_clip; this->far_clip = far_clip; }
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
	__hdfi__ float3 direction_for_pixel(float2 uv) const{
		float3 direction;
		direction.z = 1;
		direction.x = 2 * ((float)uv.x / w - 0.5f);
		direction.y = 2 * ((float)uv.y / h - 0.5f);
		//reverse the effects of projection
		direction.x /= -perspective[0][0];
		direction.y /= -perspective[1][1];
		direction.z /= perspective[2][2];
		//reverse effects of transform
		return transform.inverted_direction(direction);
	}
};
constexpr float PI = 3.14159265358979323851f;

/*
* Camera data, full struct. Used for freeCam. Just what the standard OpenGL camera is with some extra data to make moving it around easy. The only camera type that adapts to viewport size for now.
*/
struct InteractiveCameraData:CameraDataItf,PartialInteractiveCameraData {
	virtual CameraType_t type() const { return CameraType::INTERACTIVE; };
	virtual int get_width() const { return w; };
	virtual int get_height() const { return h; };
	virtual std::unique_ptr<CameraDataItf> scaleTo(int w, int h) const {
		auto scaled=std::make_unique<InteractiveCameraData>(*this); scaled->w = w; scaled->h = h;
		return scaled;
	};
	virtual std::unique_ptr<CameraDataItf> subSection(int x1, int y1, int x2, int y2) const override { throw "InteractiveCameraData::subSection is unimplemented"; };
	float scaleY,fov_x,fov_rad;
	float yaw, pitch, roll;
	bool flipped_x;
	int selected_scene=0;
	int debug_channels=3;
	int debug_channel_start=0;
	bool show_mips = false;
	float3 translation;
	InteractiveCameraData() :PartialInteractiveCameraData{},
		scaleY{ 1 }, fov_x{ 0 }, fov_rad{ 0 },
		yaw{ 0 }, pitch{ 0 }, roll{ 0 }, flipped_x{ false },
		translation{ 0,0,0 } {};
	InteractiveCameraData(float scaleY,float fov_rad,float near_clip,float far_clip,float yaw=0,float pitch=0,float roll=0,bool flipped_x=false,float3 translation=make_float3(0,0,0),int w=-1,int h=-1)
		:PartialInteractiveCameraData{ near_clip,far_clip,w,h }, 
		scaleY{ scaleY }, fov_rad{ fov_rad },
		yaw{ yaw }, pitch{ pitch }, roll{ roll }, flipped_x{ flipped_x },
		translation{ translation }
	{ recalculateTangent(); recalculateTransform(); }
	InteractiveCameraData(const InteractiveCameraData& ot) = default;
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

	void set_debug_channels(int num) { debug_channels = num; }
	int get_debug_channels() { return debug_channels; };

	void nudge_debug_channel_start(int num) {
		debug_channel_start = debug_channel_start+num;//%num_channels would usually go here, but we don't have that information here.
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
		transform = transform_from(yaw, pitch, roll, translation);
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

	PartialInteractiveCameraData prepareForGPU(int height, int width) const {
		//Intentional slicing
		PartialInteractiveCameraData data;
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
	virtual std::string serialized(bool text) const override;
	virtual ~InteractiveCameraData() override {};
};


//Pinhole projection camera data.
struct PartialPinholeCameraData :PartialCameraDataTemplate<PartialPinholeCameraData> {
	float ppx,ppy,fx,fy;

	//These should be implemented for any camera type, or else the inline thing in kernels may break.
	__hdfi__ float4 mapToWorldCoords(float4 coords) const {
		return transform * coords;
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float4 world_coords) const {
		if (world_coords.z < near_clip || world_coords.z > far_clip) return ScreenCoordsWithDepth::invalid();
		float inv_depth = 1/ world_coords.z;

		int dx = (int)(-fx * world_coords.x * inv_depth + ppx);
		int dy = (int)(-fy * world_coords.y * inv_depth + ppy);
		if (dx < 0 || dy < 0 || dx >= w || dy >= h)return ScreenCoordsWithDepth::invalid();
		return ScreenCoordsWithDepth(make_int2(dx, dy), world_coords.z);
	};
	__hdfi__ float3 direction_for_pixel(float2 uv) const {
		float3 direction;
		direction.z = 1;
		direction.x = -(uv.x - ppx) / fx;
		direction.y = -(uv.y - ppy) / fy;
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
	virtual std::unique_ptr<CameraDataItf> subSection(int x1, int y1, int x2, int y2) const override{
		auto subsection = std::make_unique<PinholeCameraData>(*this);
		subsection->w = x2 - x1; subsection->h = y2 - y1;
		subsection->ppx -= x1; subsection->ppy -= y1;
		return subsection;
	};
	PartialPinholeCameraData prepareForGPU(int height, int width) const {
		//This is intentional slicing. The only real things sliced off are v-tables and such
		return *this;
	}
	PinholeCameraData() {};
	PinholeCameraData(const PinholeCameraData& ot) = default;
	PinholeCameraData(PinholeCameraData&& ot) = default;
	virtual std::string serialized(bool text) const override;
};

//Radial projection camera data.
//Implementations more or less copied from: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
struct PartialRadialCameraData :PartialCameraDataTemplate<PartialRadialCameraData> {
	float fx, fy, ppx, ppy, k1, k2;

	//These should be implemented for any camera type, or else the inline thing in kernels may break.
	__hdfi__ float4 mapToWorldCoords(float4 coords) const {
		return transform * coords;
	};

#pragma warning(push)
#pragma warning(disable:4244) //conversion from double to float
	__hdfi__ void iterativeUndistortion(float* u, float* v) const {
		// Parameters for Newton iteration using numerical differentiation with
		// central differences, 100 iterations should be enough even for complex
		// camera models with higher order terms.
		const size_t kNumIterations = 5;
		const double kMaxStepNorm = 1e-10;
		const double kRelStepSize = 1e-6;

		float4 J;
		const float2 x0 = make_float2(*u, *v);
		float2 x = make_float2(*u, *v);
		float2 dx = make_float2(0, 0);
		float2 dx_0b = make_float2(0, 0);
		float2 dx_0f = make_float2(0, 0);
		float2 dx_1b = make_float2(0, 0);
		float2 dx_1f = make_float2(0, 0);

		for (size_t i = 0; i < kNumIterations; ++i) {
			const double step0 = std::max(std::numeric_limits<double>::epsilon(),
				std::abs(kRelStepSize * x.x));
			const double step1 = std::max(std::numeric_limits<double>::epsilon(),
				std::abs(kRelStepSize * x.y));
			distortion(x.x, x.y, &dx.x, &dx.y);
			distortion(x.x - step0, x.y, &dx_0b.x, &dx_0b.y);
			distortion(x.x + step0, x.y, &dx_0f.x, &dx_0f.y);
			distortion(x.x, x.y - step1, &dx_1b.x, &dx_1b.y);
			distortion(x.x, x.y + step1, &dx_1f.x, &dx_1f.y);
			J/*(0, 0)*/.x = 1 + (dx_0f.x - dx_0b.x) / (2 * step0);
			J/*(0, 1)*/.y = (dx_1f.x - dx_1b.x) / (2 * step1);
			J/*(1, 0)*/.z = (dx_0f.y - dx_0b.y) / (2 * step0);
			J/*(1, 1)*/.w = 1 + (dx_1f.y - dx_1b.y) / (2 * step1);
			float2 xm = make_float2(x.x + dx.x - x0.x, x.y + dx.y - x0.y);
			float2 step_x = make_float2(xm.x*J.x + xm.y*J.z, xm.x * J.y + xm.y * J.w);//correct?
			x.x = step_x.x;
			x.y = step_x.y;
			if (x.x*x.x + x.y*x.y < kMaxStepNorm) {
				break;
			}
		}

		*u = x.x;
		*v = x.y;
	}

	__hdfi__ void distortion(const float u, const float v,
		float* du, float* dv) const {
		const float u2 = u * u;
		const float v2 = v * v;
		const float r2 = u2 + v2;
		const float radial = k1 * r2 + k2 * r2 * r2;
		*du = u * radial;
		*dv = v * radial;
	}
#pragma warning(pop)

	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float4 world_coords) const {
		if (world_coords.z < near_clip || world_coords.z > far_clip) return ScreenCoordsWithDepth::invalid();
		float inv_depth = 1 / world_coords.z;
		float nx = world_coords.x * inv_depth;
		float ny = world_coords.y * inv_depth;
		float du, dv;
		distortion(nx, ny, &du, &dv);
		nx += du;
		ny += dv;

		int dx = (int)(-fx * nx + ppx);
		int dy = (int)(-fy * ny + ppy);
		if (dx < 0 || dy < 0 || dx >= w || dy >= h)return ScreenCoordsWithDepth(make_int2(dx, dy), world_coords.z, false);
		return ScreenCoordsWithDepth(make_int2(dx, dy), world_coords.z);
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float4 coords) const {
		return mapToScreenCoordsFromWorldCoords(mapToWorldCoords(coords));
	};
	__hdfi__ float3 direction_for_pixel(float2 uv) const {
		float3 direction;
		direction.z = 1;
		direction.x = -(uv.x - ppx) / fx;
		direction.y = -(uv.y - ppy) / fy;

		iterativeUndistortion(&direction.x, &direction.y);

		return transform.inverted_direction(direction);
	}
	PartialRadialCameraData() { fx = fy = k1 = k2 = ppx = ppy = near_clip = far_clip = 0; };
};

struct RadialCameraData :CameraDataItf, PartialRadialCameraData {
	virtual CameraType_t type() const { return CameraType::RADIAL; };
	virtual int get_width() const { return w; };
	virtual int get_height() const { return h; };
	virtual std::unique_ptr<CameraDataItf> scaleTo(int w, int h) const {
		auto scaled = std::make_unique<RadialCameraData>(*this); scaled->w = w; scaled->h = h;
		float dx = (1.f * w / this->w);
		float dy = (1.f * h / this->h);

		scaled->fx *= dx; scaled->fy *= dy;

		scaled->ppx *= dx; scaled->ppy *= dy;
		return scaled;
	};
	virtual std::unique_ptr<CameraDataItf> subSection(int x1, int y1, int x2, int y2) const override {
		auto subsection = std::make_unique<RadialCameraData>(*this);
		subsection->w = x2 - x1; subsection->h = y2 - y1;
		subsection->ppx -= x1; subsection->ppy -= y1;
		return subsection;
	};
	PartialRadialCameraData prepareForGPU(int height, int width) const {
		//This is intentional slicing. The only real things sliced off are v-tables and such
		return *this;
	}
	RadialCameraData() {};
	RadialCameraData(const RadialCameraData& ot) = default;
	RadialCameraData(RadialCameraData&& ot) = default;
	virtual std::string serialized(bool text) const override;
};
