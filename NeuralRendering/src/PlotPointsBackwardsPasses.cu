#include "CudaIncludesAndMacros.cuh"
#include "PlotPointsBackwardsPasses.cuh"
#include "cuda_debug_utils.cuh"

/*template <typename camera_type_partial>
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
}*/


__global__ void backwards_points(const PartialInteractiveCameraData camera, int ndim, float* point_data, float* points_gradient, int num_points, const float* plot_gradient) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        int ids = idx * (3 + ndim);
        ScreenCoordsWithDepth d = camera.mapToScreenCoords(make_float4(point_data[ids + 0], point_data[ids + 1], point_data[ids + 2], 1));
        if (d.valid) {
            int pixel = d.coords.x + d.coords.y * camera.w;
            //todo: depth test, somehow, probably by saving depth and weights somewhere.
            const float* grad_start = &plot_gradient[pixel * (ndim + 1)];
            float* point_grad_start = &(points_gradient[ids + 3]);
            for (int i = 0; i < ndim; ++i) {
                point_grad_start[i] += grad_start[i];
            }
        }
    }
}

template <typename camera_type_partial>
__global__ void bacwards_environment_v2(const camera_type_partial camera, int ndim, float* environment_grad, int environment_resolution, const float* plot, const float* plot_gradient) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < camera.h && idy < camera.w) {
        int ids = (idy + idx * camera.w) * (ndim + 1);
        if (plot[ids+ndim] < 0) {//todo: change so it makes more sense, possibly keeping track of what points are free while testing points.
            float3 direction = camera.direction_for_pixel(make_int2(idx, idy));
            unsigned int adress = pixel_from_cubemap_coords(environment_resolution, cubemap_coords(environment_resolution, direction));
            for (int i = 0; i < ndim; ++i) {
                atomicAdd(&environment_grad[adress+i], plot_gradient[ids+i]);
            }
            //atomicAdd(&environment_grad[adress + ndim], plot_gradient[ids + ndim]); //depth is a little more complicated.
        }
    }
}


template <typename camera_type_partial>
cudaError_t backwards_environment_for_camera_v2(const camera_type_partial& camera, int ndim, 
    const void* environment, void* environment_grad, int environment_resolution,
    const void* plot, const void* plot_gradient) {
    cudaError_t cudaStatus;
    bacwards_environment_v2 BEST_2D_KERNEL(camera.h, camera.w) (camera, ndim, (float*)environment_grad, environment_resolution, (float*)plot, (float*)plot_gradient);
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();
Error:
    return cudaStatus;
}


cudaError_t PlotPointsBackwardsPass_v2(const void* point_data, void* point_grad, int num_points,
    const void* environment, void* environment_grad, int environment_resolution,
    const std::shared_ptr<CameraDataItf> camera, int h, int w, int ndim,
    const void* plot, const void* plot_grad)
{
    cudaError_t cudaStatus;
    if (h < 0)h = camera->get_height();
    if (w < 0)w = camera->get_width();
    switch (camera->type())
    {
    case INTERACTIVE:
        cudaStatus = backwards_environment_for_camera_v2(((InteractiveCameraData&)(*camera)).prepareForGPU(w, h), ndim, environment, environment_grad, environment_resolution, plot, plot_grad);
        STATUS_CHECK();
        break;
    case PINHOLE_PROJECTION:
        cudaStatus = backwards_environment_for_camera_v2(((PinholeCameraData&     )(*camera)).prepareForGPU(w,h), ndim, environment, environment_grad, environment_resolution, plot, plot_grad);
        STATUS_CHECK();
        break;
    default:
        std::cerr << "INVALID CAMERA FOUND AT " << __LINE__ << " of " << __FILE__ << '\n';
        break;
    }
Error:
    //TODO: implement
    return cudaStatus;
}
