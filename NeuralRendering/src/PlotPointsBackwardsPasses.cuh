#pragma once
#include <vector>
#include "cuda.h"
#include "Points.hpp"
#include "CameraData.hpp"
#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"


cudaError_t PlotPointsBackwardsPass_v2(const std::shared_ptr<CameraDataItf> camera, int ndim, 
    const void* point_data, void* point_grad, int num_points,
    const void* environment, void* environment_grad, int environment_resolution,
    const void* plot, const void* plot_weights, const void* plot_grad);
