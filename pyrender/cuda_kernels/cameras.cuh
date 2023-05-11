#include <cuda_runtime.h>
//No reason to include __host__
#define __hdfi__ __device__ __forceinline__
struct CAMERA_DATA_TYPE;

//This is a better abstraction than a 4x4 matrix, and its functions are simpler and more explicit.
struct Transform{
    float rotation[9];
    float translation[3];
    __hdfi__ Transform(){
        rotation[0]=1;rotation[1]=0;rotation[2]=0;
        rotation[3]=0;rotation[4]=1;rotation[5]=0;
        rotation[6]=0;rotation[7]=0;rotation[8]=1;
    }
    //NOTE: rotation matrix is multiplied to the left.
    __hdfi__ float3 apply_rotation(float3 pos) const{
        float3 out;
        out.x = pos.x*rotation[0] + pos.y*rotation[3] + pos.z*rotation[6];
        out.y = pos.x*rotation[1] + pos.y*rotation[4] + pos.z*rotation[7];
        out.z = pos.x*rotation[2] + pos.y*rotation[5] + pos.z*rotation[8];
        return out;
    }
    __hdfi__ float3 unapply_rotation(float3 pos) const{
        float3 out;
        //The inverse of an orthogonal matrix is its transpose. source: https://en.wikipedia.org/wiki/Orthogonal_matrix
        //Any rotation matrix is orthogonal, because the unit vector in each direction remains unitary and perpendicular to the others after transformation.
        out.x = pos.x*rotation[0] + pos.y*rotation[1] + pos.z*rotation[2];
        out.y = pos.x*rotation[3] + pos.y*rotation[4] + pos.z*rotation[5];
        out.z = pos.x*rotation[6] + pos.y*rotation[7] + pos.z*rotation[8];
        return out;
    }
    __hdfi__ float3 apply_translation(float3 pos) const{
        float3 out;
        out.x = pos.x+translation[0];
        out.y = pos.y+translation[1];
        out.z = pos.z+translation[2];
        return out;
    }
    __hdfi__ float3 unapply_translation(float3 pos) const{
        float3 out;
        out.x = pos.x-translation[0];
        out.y = pos.y-translation[1];
        out.z = pos.z-translation[2];
        return out;
    }

    __hdfi__ float3 apply(float3 pos) const{
        return apply_rotation(apply_translation(pos));
    }
    __hdfi__ float3 unapply(float3 pos) const{
        return unapply_translation(unapply_rotation(pos));
    }
    __hdfi__ static Transform from(const float* data){
		Transform ret;
#pragma unroll
        for(int i=0;i<9;++i)ret.rotation[i]   =data[i];
#pragma unroll
        for(int i=0;i<3;++i)ret.translation[i]=data[9+i];
		return ret;
    }
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
		return ScreenCoordsWithDepth(make_int2(0, 0), 0.0, false);
	}
};

struct PartialCameraDataTemplate
{
	float w0,h0,w,h;
	Transform transform = Transform();
	float near_clip = 0, far_clip = 1e20f;

	__hdfi__ PartialCameraDataTemplate() { w0=h=h=w=0; };
};


#if CAM_TYPE==0 && !defined(CAMERA_DATA_TYPE)
#define CAMERA_DATA_TYPE PinholeCameraData
struct PinholeCameraData:PartialCameraDataTemplate{
	float ppx,ppy,fx,fy,lum;
	//These should be implemented for any camera type, or else the inline thing in kernels may break.
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float3 world_coords) const {
		if (world_coords.z < near_clip || world_coords.z > far_clip) return ScreenCoordsWithDepth::invalid();
		float inv_depth = 1/ world_coords.z;

		int dx = lroundf(-fx * world_coords.x * inv_depth + ppx - w0);
		int dy = lroundf(-fy * world_coords.y * inv_depth + ppy - h0);
		if (dx < 0 || dy < 0 || dx >= w || dy >= h)return ScreenCoordsWithDepth::invalid();
		return ScreenCoordsWithDepth(make_int2(dx, dy), world_coords.z);
	};
	__hdfi__ float3 direction_for_pixel(float2 uv) const {
		float3 direction;
		direction.z = 1;
		direction.x = -(uv.x+w0 - ppx) / fx;
		direction.y = -(uv.y+h0 - ppy) / fy;
		return transform.unapply_rotation(direction);
	}
	__hdfi__ PinholeCameraData(const float* data) {
        w0 =data[0];
        h0 =data[1];
        w  =data[2];
        h  =data[3];
        lum=data[4];
        transform=Transform::from(data+5);
        ppx=data[5+9+3+0];
        ppy=data[5+9+3+1];
        fx =data[5+9+3+2];
        fy =data[5+9+3+3];
    };
};
#endif

#if CAM_TYPE==1 && !defined(CAMERA_DATA_TYPE)
#define CAMERA_DATA_TYPE RadialCameraData
//Radial projection camera data.
//Implementations more or less copied from: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
struct PartialRadialCameraData :PartialCameraDataTemplate<PartialRadialCameraData> {
	float fx, fy, ppx, ppy, k1, k2;


    __hdfi__ PartialRadialCameraData(const float* data){
        w0 =data[0];
        h0 =data[1];
        w  =data[2];
        h  =data[3];
        lum=data[4];
        transform=Transform::from(data+5);
        ppx=data[5+9+3+0];
        ppy=data[5+9+3+1];
        fx =data[5+9+3+2];
        fy =data[5+9+3+3];
        k1 =data[5+9+3+4];
        k2 =data[5+9+3+5];
    }


	//These should be implemented for any camera type, or else the inline thing in kernels may break.
	__hdfi__ float4 mapToWorldCoords(float4 coords) const {
		return transform * coords;
	};

	__hdfi__ void iterativeUndistortion(float* u, float* v) const {
		// Parameters for Newton iteration using numerical differentiation with
		// central differences, 100 iterations should be enough even for complex
		// camera models with higher order terms.
		const size_t kNumIterations = 5;//100;
		const float kMaxStepNorm = 1e-10;
		const float kRelStepSize = 1e-6;

		float4 J;
		const float2 x0 = make_float2(*u, *v);
		float2 x = make_float2(*u, *v);
		float2 dx = make_float2(0.0f, 0.0f);
		float2 dx_0b = make_float2(0.0f, 0.0f);
		float2 dx_0f = make_float2(0.0f, 0.0f);
		float2 dx_1b = make_float2(0.0f, 0.0f);
		float2 dx_1f = make_float2(0.0f, 0.0f);

		for (size_t i = 0; i < kNumIterations; ++i) {
			const float step0 = std::max(std::numeric_limits<float>::epsilon(),
				std::abs(kRelStepSize * x.x));
			const float step1 = std::max(std::numeric_limits<float>::epsilon(),
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

	__hdfi__ ScreenCoordsWithDepth mapToScreenCoordsFromWorldCoords(float3 world_coords) const {
		if (world_coords.z < near_clip || world_coords.z > far_clip) return ScreenCoordsWithDepth::invalid();
		float inv_depth = 1 / world_coords.z;
		float nx = world_coords.x * inv_depth;
		float ny = world_coords.y * inv_depth;
		float du, dv;
		distortion(nx, ny, &du, &dv);
		nx += du;
		ny += dv;

		int dx = lroundf(-fx * nx + ppx - w0);
		int dy = lroundf(-fy * ny + ppy - h0);
		if (dx < 0 || dy < 0 || dx >= w || dy >= h)return ScreenCoordsWithDepth::invalid();
		return ScreenCoordsWithDepth(make_int2(dx, dy), world_coords.z);
	};
	__hdfi__ float3 direction_for_pixel(float2 uv) const {
		float3 direction;
		direction.z = 1;
		direction.x = -(uv.x - ppx) / fx;
		direction.y = -(uv.y - ppy) / fy;

		iterativeUndistortion(&direction.x, &direction.y);

		return transform.unapply_rotation(direction);
	}
	PartialRadialCameraData() { fx = fy = k1 = k2 = ppx = ppy = near_clip = far_clip = 0; };
};
#endif






struct Camera:CAMERA_DATA_TYPE{
	__hdfi__ Camera(const float* data):CAMERA_DATA_TYPE{data} {}
	__hdfi__ void implementations_exist() {
		//makes sure these are all required functions exist, yet should be optimized away.
		mapToScreenCoords(direction_for_pixel(make_float2(0, 0)));
	}
	__hdfi__ float3 mapToWorldCoords(float3 coords) const {
		return transform.apply(coords);
	};
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float4 coords) const {
		return mapToScreenCoords(make_float3(coords.x, coords.y, coords.z));
	}
	__hdfi__ ScreenCoordsWithDepth mapToScreenCoords(float3 coords) const {
		return mapToScreenCoordsFromWorldCoords(mapToWorldCoords(coords));
	}
};
