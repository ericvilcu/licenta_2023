#include "CudaIncludesAndMacros.cuh"
#include "CudaMath.cuh"
#include "cuda_debug_utils.cuh"
#define DEFAULT_SET_STEP 1
template <typename T, int step>
__global__ void set_all(T* data, T value, int mx) {
	//todo: delete.
	int idx = (threadIdx.x + blockDim.x * blockIdx.x)*step;
	if (step == 1) {
		if (idx < mx) {
			data[idx] = value;
		}
	}
	else {
		for (int d = 0; d < step; ++d) {
			if (idx+d < mx) {
				data[idx+d] = value;
			}
		}
	}
}

template <typename T, int step=1>
__global__ void add(T* data_out,const T* data_in, size_t mx) {
	//todo: delete.
	int idx = (threadIdx.x + blockDim.x * blockIdx.x) * step;
	if (step == 1) {
		if (idx < mx) {
			data_out[idx] += data_in[idx];
		}
	}
	else {
		for (int d = 0; d < step; ++d) {
			if (idx + d < mx) {
				data_out[idx + d] += data_in[idx + d];
			}
		}
	}
}

//should not really be used...
//I need to find some way to store my colors as floats.
__global__ void nudge_bytes(unsigned char* dst, const float* src, size_t mx, float inv_treshold) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < mx) {
		dst[idx] = (unsigned int)round(clamp(dst[idx] + clamp(src[idx]*inv_treshold, -1.0f, 1.0f),0.0f,255.0f));
	}
}

cudaError_t set_all_to_zeros(void* memory, size_t num_bytes)
{
	return cudaMemset(memory, 0, num_bytes);
}

cudaError_t apply_gradient(void* vec, const void* grad, size_t num_items, bool dest_is_bytes)
{
	cudaError_t cudaStatus = cudaSuccess;
	if (dest_is_bytes) {
		nudge_bytes BEST_LINEAR_KERNEL(num_items) ((unsigned char*)vec, (const float*)grad, num_items,1/1e-8);
		cudaStatus = cudaDeviceSynchronize();
		STATUS_CHECK();
		cudaStatus = cudaPeekAtLastError();
		STATUS_CHECK();
	}
	else {
		add BEST_LINEAR_KERNEL(num_items) ((float*)vec, (const float*)grad, num_items);
		cudaStatus = cudaDeviceSynchronize();
		STATUS_CHECK();
		cudaStatus = cudaPeekAtLastError();
		STATUS_CHECK();
	}
	Error:
	return cudaStatus;
}
