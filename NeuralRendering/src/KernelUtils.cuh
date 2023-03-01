#pragma once
#include <math.h>
#include "CudaIncludesAndMacros.cuh"
#define approx_PI 3.141592
//Essentially, __utilf means: 1. always inline, and 2. can work on CPU *and* GPU
#define __utilf __device__ __host__ __forceinline__

template <typename T>
__utilf T clamp(T x, T mn, T mx) {
	if (x < mn) return mn;
	if (x > mx) return mx;
	return x;
}

template <typename T, typename M>
__utilf T mix(T f, T s, M amt) {
	return (T)(f * (1 - amt) + s * amt);
}
template <typename T, typename M>
__utilf T mixc(T f, T s, M amt) {
	return mix(f, s, clamp<M>(amt, 0, 1));
}

//Using templates for everything seems tough.
//Unless I find a way to use decltype to generalize them further as to not even need abs# and such, I will just write whatever I need whenever I need it here.

template<typename T>
__utilf T abs(T s) {
	if (s > 0) return s;
	return -s;
}
template<typename T2>
__utilf T2 abs2(T2 s) {
	T2 out;
	out.x = abs(s.x);
	out.y = abs(s.y);
	return out;
}
template<typename T3>
__utilf T3 abs3(T3 s) {
	//decltype(T3::x);
	//decltype(T3::y);
	//decltype(T3::z);
	T3 out;
	out.x = abs(s.x);
	out.y = abs(s.y);
	out.z = abs(s.z);
	return out;
}
template<typename T4>
__utilf T4 abs4(T4 s) {
	T4 out;
	out.x = abs(s.x);
	out.y = abs(s.y);
	out.z = abs(s.z);
	out.w = abs(s.w);
	return out;
}

__utilf float3 add(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__utilf float frac(float in) {
	//return fmodf(in, 1.0f);
	return in - floorf(in);
}

__utilf float3 frac(float3 in) {
	return make_float3(frac(in.x), frac(in.y), frac(in.z));
}

__utilf float4 frac(float4 in) {
	return make_float4(frac(in.x), frac(in.y), frac(in.z), frac(in.w));
}


__utilf uchar4 toRGBA8(float4 rgba) {
	return make_uchar4((unsigned char)(255.0f * rgba.x), (unsigned char)(255.0f * rgba.y), (unsigned char)(255.0f * rgba.z), (unsigned char)(255.0f * rgba.w));
}

__utilf float4 toRGBA32f(uchar4 rgba) {
	return make_float4(1 / 255.0f * rgba.x, 1 / 255.0f * rgba.y, 1 / 255.0f * rgba.z, 1 / 255.0f * rgba.w);
}



__utilf float4 fromHSV(float4 hsva) {
	/*
	Original HLSL code:
	float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	float3 p = abs(frac(input.xxx + K.xyz) * 6.0 - K.www);

	return input.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), input.y);

	Cuda NEEDS a library for these things. It's absurd.
	*/
	float3 p = frac(make_float3(hsva.x + 1.0f, hsva.x + 2.0f / 3.0f, hsva.x + 1.0f / 3.0f));
	p.x = abs(p.x * 6.0f - 3.0f);
	p.y = abs(p.y * 6.0f - 3.0f);
	p.z = abs(p.z * 6.0f - 3.0f);
	float4 rgba;
	rgba.x = mix(1.0f, clamp(p.x - 1.0f, 0.0f, 1.0f), hsva.y) * hsva.z;
	rgba.y = mix(1.0f, clamp(p.y - 1.0f, 0.0f, 1.0f), hsva.y) * hsva.z;
	rgba.z = mix(1.0f, clamp(p.z - 1.0f, 0.0f, 1.0f), hsva.y) * hsva.z;
	rgba.w = hsva.w;
	return rgba;
}


__utilf float magnitude(float3 v) {
	return sqrtf((v.x*v.x)+(v.y*v.y)+(v.z*v.z));
}

__utilf float3 normalized(float3 v) {
	float mag = magnitude(v);
	return make_float3(v.x, v.y, v.z);
}

__utilf float4 float4_from(uchar4 rgba) {
	return make_float4(rgba.x * (1 / 255.0f), rgba.y * (1 / 255.0f), rgba.z * (1 / 255.0f), rgba.w * (1 / 255.0f));
}


__utilf uint3 cubemap_coords(int resolution, float3 direction) {
	float3 abs_dir = abs3(direction);
	//rewrite of: https://www.gamedev.net/forums/topic/687535-implementing-a-cube-map-lookup-function/5337472/
	if (abs_dir.z >= abs_dir.y && abs_dir.z >= abs_dir.x) {
		//Positive/negative Z
		int face_idx = (direction.z > 0 ? 1 : 0);
		float D = 0.5f / abs_dir.z;
		uint2 uv = make_uint2((unsigned int)clamp<float>((direction.x*D+0.5f)*resolution,0.0f,(float)(resolution-1)),
							  (unsigned int)clamp<float>((direction.y*D+0.5f)*resolution,0.0f,(float)(resolution-1)));
		return make_uint3(face_idx, uv.x, uv.y);
	}
	else if (abs_dir.y >= abs_dir.x) {
		//Positive/negative y
		int face_idx = (direction.y > 0 ? 3 : 2);
		float D = 0.5f / abs_dir.y;
		uint2 uv = make_uint2((unsigned int)clamp<float>((direction.x*D+0.5f)*resolution,0.0f,(float)(resolution-1)),
							  (unsigned int)clamp<float>((direction.z*D+0.5f)*resolution,0.0f,(float)(resolution-1)));
		return make_uint3(face_idx, uv.x, uv.y);
	}
	else {
		//Positive/negative x
		int face_idx = (direction.x > 0 ? 5 : 4);
		float D = 0.5f / abs_dir.x;
		uint2 uv = make_uint2((unsigned int)clamp<float>((direction.z*D+0.5f)*resolution,0.0f,(float)(resolution-1)),
							  (unsigned int)clamp<float>((direction.y*D+0.5f)*resolution,0.0f,(float)(resolution-1)));
		return make_uint3(face_idx, uv.x, uv.y);
	}
}
__utilf unsigned int pixel_from_cubemap_coords(int resolution, uint3 data) {
	//clamps may be unnecesary
	return (unsigned int)(data.x * resolution * resolution + clamp<unsigned int>(data.y, 0, resolution - 1) + clamp<unsigned int>(data.z, 0, resolution - 1) * resolution);
}
//Plotting stuff
__utilf float4 sample_environment_data(uchar4* environment_data, int resolution, float3 direction) {
	return float4_from(environment_data[pixel_from_cubemap_coords(resolution, cubemap_coords(resolution, direction))]);
}
__utilf float4 sample_environment_data(float4* environment_data, int resolution, float3 direction) {
	return environment_data[pixel_from_cubemap_coords(resolution, cubemap_coords(resolution, direction))];
}

__utilf float* sample_environment_data_v2(float* environment_data, int resolution, float3 direction, int ndim) {
	return &environment_data[(ndim+1) * pixel_from_cubemap_coords(resolution, cubemap_coords(resolution, direction))];
}

