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
* @brief Plots to GPU memory. May return a depth buffer if it is not null
*
* @param memory GPU memory, preallocated or not. specify with the is_preallocated parameter.
*/
cudaError_t plotPointsToGPUMemory_v2(const std::shared_ptr<CameraDataItf> camera, int ndim,
    const void* point_memory, int num_points,
    const void* environment_memory, int environment_resolution,
    float** memory_color, bool is_preallocated = false,
    float** memory_weights = NULL, bool is_weight_preallocated = false,
    bool needs_clear = true);

inline cudaError_t plotPointsToGPUMemory_preallocated_v2(const std::shared_ptr<CameraDataItf> camera, int ndim,
    const void* point_memory, int num_points, const void* environment_memory, int environment_resolution,
    float* memory_color, float* memory_weights = NULL) {
    return plotPointsToGPUMemory_v2(camera, ndim,
        point_memory, num_points, environment_memory, environment_resolution,
        &memory_color, true, (memory_weights == NULL ? NULL : &memory_weights), memory_weights != NULL);
}

cudaError_t to_CPU(void*& memory, int length);
cudaError_t to_GPU(void*& memory, int length);