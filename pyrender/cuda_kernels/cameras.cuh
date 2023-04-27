#include <cuda_runtime.h>
//No reason to include __host__
#define __hdfi__ __device__ __forceinline__
#if CAM_TYPE==0 && !defined(CAMERA_DATA_TYPE)
#define CAMERA_DATA_TYPE PinholeCameraData
#endif
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
