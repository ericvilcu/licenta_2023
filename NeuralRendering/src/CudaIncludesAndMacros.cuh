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
#include "KernelUtils.cuh"
#include "cuda_texture_types.h"
#ifdef __INTELLISENSE__
    #include "intellisense_cuda_intrinsics.cuh"
#endif


#ifndef NO_STATUS_CHECK
#define STATUS_CHECK()do{if(cudaStatus!=cudaSuccess){std::cerr<<"Cuda failure at line:"<<__LINE__<<";File:"<<__FILE__<<";Code:"<<cudaStatus<<" String:"<< cudaGetErrorString ( cudaStatus ) <<"\n";goto Error;}}while(0)
#else // DEBUG
#define STATUS_CHECK()
#endif
#define CHK STATUS_CHECK();

//Hacks stolen from:
//https://stackoverflow.com/questions/6061565/setting-up-visual-studio-intellisense-for-cuda-kernel-calls
//https://forums.developer.nvidia.com/t/includes-for-surf2dwrite/39038/2
//This is to prevent intellisense from showing an error whenever the <<<...>>> syntax is used.
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...) 
#define CUDA_TEXTURE(...) textureReference
template<typename T>
void surf2Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y,
    cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define CUDA_TEXTURE(...) texture<__VA_ARGS__>
#endif
//Note: currently not the best 2d/linear kernels to choose
#define default_max_kernel 1024
#define default_max_kenrel_sqrt 32
#define BEST_2D_KERNEL(H,W) CUDA_KERNEL(dim3((int)(((H)+(default_max_kenrel_sqrt-1))/default_max_kenrel_sqrt),(int)(((W)+(default_max_kenrel_sqrt-1))/default_max_kenrel_sqrt),1),dim3(default_max_kenrel_sqrt,default_max_kenrel_sqrt,1))
#define BEST_LINEAR_KERNEL(D) CUDA_KERNEL((((unsigned int)D)+(default_max_kernel-1))/default_max_kernel,default_max_kernel)