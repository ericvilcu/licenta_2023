#pragma once
#include "cuda.h"

#if WIN32
#include <Windows.h>
#endif
#include "cuda_gl_interop.h"

#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"

#include "device_launch_parameters.h"
#include <iostream>
//#include "cudaGLTypedefs.h"
//#include "cudaGL.h"
#include "cuda_texture_types.h"
#ifdef __INTELLISENSE__
    #include "intellisense_cuda_intrinsics.cuh"
#endif

#include "KernelUtils.cuh"

#if defined(_DEBUG) || TRUE
//TODO: pass cudaStatus as an argument to the macro to stop it being stupid
#define MSG_CHECK(MSG) do{if(cudaStatus!=cudaSuccess){std::cerr<<MSG<<"Cuda failure at line:"<<__LINE__<<";File:"<<__FILE__<<";Code:"<<cudaStatus<<" String:"<< cudaGetErrorString ( cudaStatus ) <<"\n";goto Error;}}while(0)
#define STATUS_CHECK() MSG_CHECK("")
#define AFTER_FUNCTION_CALL_CHECK() do{cudaStatus = cudaDeviceSynchronize();MSG_CHECK("(at cudaDeviceSynchronize)");cudaStatus = cudaPeekAtLastError();MSG_CHECK("(at cudaPeekAtLastError)");}while(0);
//unneccesary if AFTER_FUNCTION_CALL_CHECK is used every time as well.


//TODO: pass cudaStatus as an argument to the macro to stop it being stupid
//would this work?
#define CHECK_ERROR(cudaStatus) ([](cudaError_t status){if(status!=cudaSuccess){std::cerr<<MSG<<"Cuda failure at line:"<<__LINE__<<";File:"<<__FILE__<<";Code:"<<cudaStatus<<" String:"<< cudaGetErrorString ( cudaStatus ) <<"\n";return true;}return false;})(cudaStatus)
#define ENSURE_SYNC()
#else // DEBUG
#define NO_STATUS_CHECK
#define MSG_CHECK(...)
#define STATUS_CHECK()
#define CHECK_ERROR(...) false
#define AFTER_FUNCTION_CALL_CHECK()
#define ENSURE_SYNC() cudaDeviceSynchronize()
#endif

//Hacks stolen from:
//https://stackoverflow.com/questions/6061565/setting-up-visual-studio-intellisense-for-cuda-kernel-calls
//https://forums.developer.nvidia.com/t/includes-for-surf2dwrite/39038/2
//This is to prevent intellisense from showing an error whenever the <<<...>>> syntax is used.
#ifdef __INTELLISENSE__
typedef unsigned char uchar;
#define CUDA_KERNEL(...) 
#define CUDA_TEXTURE(...) textureReference
template<typename T>
void surf2Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y,
    cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
#else
#define uchar unsigned char
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define CUDA_TEXTURE(...) texture<__VA_ARGS__>
#endif
//Note: currently not the best 2d/linear kernels to choose
#define default_max_kernel 1024
#define default_max_kenrel_sqrt 32
#define BEST_2D_KERNEL(H,W) CUDA_KERNEL(dim3((int)(((H)+(default_max_kenrel_sqrt-1))/default_max_kenrel_sqrt),(int)(((W)+(default_max_kenrel_sqrt-1))/default_max_kenrel_sqrt),1),dim3(default_max_kenrel_sqrt,default_max_kenrel_sqrt,1))
#define BEST_LINEAR_KERNEL(D) CUDA_KERNEL((((unsigned int)D)+(default_max_kernel-1))/default_max_kernel,default_max_kernel)