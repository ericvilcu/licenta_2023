#pragma once
#include <vector>
#include "cuda.h"
#include "CameraData.hpp"

#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"

#include "dtype4x4.cuh"
#include "Points.hpp"
#include "Renderer.hpp"


/**
* @brief Transfers bytes to a OpenGL surface.
* 
* @param memory GPU memory, containing uchar4's.
*/
cudaError_t bytesToView(const void*& memory, const int h, const int w, Renderer& renderer, Renderer::ViewType vt);
/**
* @brief Plots to GPU memory.
*
* @param memory GPU memory, preallocated or not. specify with the is_preallocated parameter.
*/
//cudaError_t plotPointsToGPUMemory(void*& memory, int h, int w, const GPUPoints& points, const CameraDataItf& camera, bool is_preallocated=false, bool as_bytes=true);

cudaError_t plotPointsToGPUMemory_v2(void*& memory, int h, int w, int ndim, const void* points_memory, int num_points, const void* environment_memory, int environment_resolution, const std::shared_ptr<CameraDataItf> camera, bool is_preallocated = false);

//cudaError_t plotPoints(const cudaSurfaceObject_t output, const int h, const int w, const GPUPoints& points, const CameraDataItf& camera);

//cudaError_t plotPointsToRenderer(const Renderer& renderer, const GPUPoints& points, const CameraDataItf& camera, Renderer::ViewType vt = Renderer::ViewTypeEnum::MAIN_VIEW);

cudaError_t to_CPU(void*& memory, int length);