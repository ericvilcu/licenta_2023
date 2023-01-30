#pragma once
#include <vector>
#include "cuda.h"
#include "Points.hpp"
#include "CameraData.hpp"
#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"


cudaError_t PlotPointsBackwardsPass(GPUPoints& points, const CameraDataItf& camera, int h, int w, const void* image_gradient, const void* input_image, float lr, bool environment);