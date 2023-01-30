#pragma once
#include <vector>
#include "cuda.h"

#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"

cudaError_t set_all_to_zeros(void* memory, size_t num_bytes);
cudaError_t apply_gradient(void* vec, const void* grad, size_t num_items, bool dest_is_bytes = true);