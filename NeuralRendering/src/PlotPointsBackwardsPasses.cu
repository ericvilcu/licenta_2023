#include "CudaIncludesAndMacros.cuh"
#include "PlotPointsBackwardsPasses.cuh"
#include "cuda_debug_utils.cuh"

template <typename camera_type_partial>
__global__ void bacwards_environment(const camera_type_partial camera, int resolution, int h, int w, float4* plot_gradient, float4* gradient_out, float4* plot, float factor) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        int ids = idy + idx * w;
        if (plot[ids].w < 0) {//todo: change so it makes more sense.
            float3 direction = camera.direction_for_pixel(make_int2(idx, idy));
            unsigned int adress = pixel_from_cubemap_coords(resolution, cubemap_coords(resolution, direction));
            //dbg[ids] = adress;
            atomicAdd(&gradient_out[adress].x, factor * plot_gradient[ids].x);
            atomicAdd(&gradient_out[adress].y, factor * plot_gradient[ids].y);
            atomicAdd(&gradient_out[adress].z, factor * plot_gradient[ids].z);
            //atomicAdd(&gradient_out[adress].w, factor * plot_gradient[ids].w); //depth is a little more complicated.
        }
    }
}

template <typename camera_type_partial>
cudaError_t backwards_environment_for_camera(const camera_type_partial& camera, Environment& environment, const void* image_gradient, const void* gradient, const void* input_image, float lr) {
    cudaError_t cudaStatus;
    //int* dbg; cudaMalloc(&dbg, camera.h * camera.w * sizeof(int));
    bacwards_environment BEST_2D_KERNEL(camera.h, camera.w) (camera, environment.resolution, camera.h, camera.w, (float4*)image_gradient, (float4*)gradient, (float4*)input_image, lr);
    //printGPUMemory<int>(dbg, camera.h * camera.w, camera.w);cudaFree(dbg);
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();
Error:
    return cudaStatus;
}


//For now, this function can only really do the environment map.
cudaError_t PlotPointsBackwardsPass(GPUPoints& points, const CameraDataItf& camera, int h, int w, const void* image_gradient, const void* input_image, float lr, bool environment)
{
    cudaError_t cudaStatus = cudaSuccess;
    if (h < 0)h = camera.get_height();
    if (w < 0)w = camera.get_width();
    if (environment) {
        points.ensure_environment();
        cudaStatus = points.environment->ensure_grad();
        STATUS_CHECK();
        switch (camera.type()) {
        case GL:
            cudaStatus=backwards_environment_for_camera(((CameraGLData&     )camera).prepareForGPU(h, w),*points.environment, image_gradient, points.environment->grad, input_image, lr);
            STATUS_CHECK();
            break;
        case PINHOLE_PROJECTION:
            cudaStatus=backwards_environment_for_camera(((PinholeCameraData&)camera).prepareForGPU(h, w),*points.environment, image_gradient, points.environment->grad, input_image, lr);
            STATUS_CHECK();
            break;
        }

    }
    //Todo: point positioning/color gradient descent

 Error:
    return cudaStatus;
}
